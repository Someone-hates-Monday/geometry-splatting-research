#!/usr/bin/env bash
# Phase-2: NCC-core recipe and optional correlation-weighted NCC.
#
# Locked recipe from G-GS ablations:
#   - lambda_depth_normal = 0
#   - lambda_multi_view_geo = 0
#   - lambda_multi_view_ncc = 0.6 (default)
#
# Variants (GGS_PHASE2_VARIANTS, comma-separated; default both):
#   ncc_core  — recipe only
#   ncc_corr  — recipe + --use_ncc_corr_weight (per-patch cc^power on ncc_loss)
#
# Pass / fail (docs plan):
#   mean overall(ncc_corr) − mean overall(ncc_core) <= -5e-4
#   and mean s2d does not get worse.
#
# Usage:
#   ./scripts/run_ggs_phase2_ncc.sh scan24 scan105
#   GGS_PHASE2_VARIANTS=ncc_core ./scripts/run_ggs_phase2_ncc.sh scan24
#
# Soft-stop:
#   ./scripts/request_train_stop.sh ./output_compare/ggs_ncc_core/scan24
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

if [ -z "${UGGS_DTU_ROOT:-}" ]; then
  echo "ERROR: UGGS_DTU_ROOT unset."
  exit 1
fi

PYTHON="${PYTHON:-python}"
BASELINE_ROOT="${BASELINE_ROOT:-./output_compare/ggs_baseline}"
CORE_ROOT="${CORE_ROOT:-./output_compare/ggs_ncc_core}"
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
ABLATION_SKIP_TRAIN="${ABLATION_SKIP_TRAIN:-0}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-$(seq 10000 1000 30000 | tr '\n' ' ')}"
GGS_PHASE2_VARIANTS="${GGS_PHASE2_VARIANTS:-ncc_core,ncc_corr}"
NCC_CORR_POWER="${NCC_CORR_POWER:-1.0}"
LAMBDA_NCC="${LAMBDA_NCC:-0.6}"

if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=(scan24 scan105)
fi

mkdir -p "$METRIC_ROOT"
chmod +x "$SCRIPT_DIR/request_train_stop.sh" 2>/dev/null || true

IFS=',' read -r -a VARIANT_LIST <<< "$GGS_PHASE2_VARIANTS"
has_variant() {
  local want="$1" v
  for v in "${VARIANT_LIST[@]}"; do
    [ "$v" = "$want" ] && return 0
  done
  return 1
}

run_train_one() {
  local sub="$1"
  local scan="$2"
  shift 2
  local out="./output_compare/$sub/$scan"
  echo "----- train $scan -> $out ($*) -----"
  # shellcheck disable=SC2086
  set +e
  "$PYTHON" train.py \
    -s "$UGGS_DTU_ROOT/$scan" \
    -m "$out" \
    -r 2 \
    --use_decoupled_appearance 3 \
    --checkpoint_iterations $CHECKPOINT_ITERS \
    --lambda_depth_normal 0 \
    --lambda_multi_view_geo 0 \
    --lambda_multi_view_ncc "$LAMBDA_NCC" \
    "$@" \
    ${START_CHECKPOINT:+--start_checkpoint "$START_CHECKPOINT"}
  local rc=$?
  set -e
  START_CHECKPOINT=""
  if [ "$rc" -eq 2 ]; then
    echo "[phase2] soft-stop at $out — skip mesh_extract."
    exit 0
  elif [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
  "$PYTHON" mesh_extract.py -m "$out"
}

compare_pair() {
  local base="$1"
  local opt="$2"
  local tag="$3"
  local scan="$4"
  local json="$METRIC_ROOT/${scan}_${tag}.json"
  if [ ! -f "$opt/recon_post.ply" ]; then
    echo "SKIP compare (no recon_post): $opt"
    return 0
  fi
  if [ ! -d "$base" ]; then
    echo "SKIP compare (missing base): $base"
    return 0
  fi
  echo "----- compare $tag $scan -----"
  "$PYTHON" scripts/compare_models.py \
    --baseline "$base" \
    --optimized "$opt" \
    --dtu_eval_root "$DTU_EVAL_ROOT" \
    --out "$json"
}

for scan in "${SCANS[@]}"; do
  if [ ! -d "$BASELINE_ROOT/$scan" ]; then
    echo "ERROR: missing $BASELINE_ROOT/$scan"
    exit 1
  fi
done

echo "[phase2] variants=$GGS_PHASE2_VARIANTS scans=${SCANS[*]} ncc=$LAMBDA_NCC corr_power=$NCC_CORR_POWER"

if [ "$ABLATION_SKIP_TRAIN" != "1" ]; then
  if has_variant ncc_core; then
    for scan in "${SCANS[@]}"; do
      echo ""
      echo "========== Phase-2 control: NCC-core (no DN, no geo) — $scan =========="
      run_train_one "ggs_ncc_core" "$scan"
    done
  fi
  if has_variant ncc_corr; then
    for scan in "${SCANS[@]}"; do
      echo ""
      echo "========== Phase-2: NCC + corr weight — $scan =========="
      run_train_one "ggs_ncc_corr" "$scan" \
        --use_ncc_corr_weight \
        --ncc_corr_power "$NCC_CORR_POWER"
    done
  fi
fi

for scan in "${SCANS[@]}"; do
  has_variant ncc_core && compare_pair \
    "$BASELINE_ROOT/$scan" "./output_compare/ggs_ncc_core/$scan" \
    "baseline_vs_ggs_ncc_core" "$scan"
  has_variant ncc_corr && compare_pair \
    "$BASELINE_ROOT/$scan" "./output_compare/ggs_ncc_corr/$scan" \
    "baseline_vs_ggs_ncc_corr" "$scan"
  # Head-to-head: corr vs core (pass criterion)
  if has_variant ncc_core && has_variant ncc_corr; then
    compare_pair \
      "./output_compare/ggs_ncc_core/$scan" "./output_compare/ggs_ncc_corr/$scan" \
      "ncc_core_vs_ggs_ncc_corr" "$scan"
  fi
done

echo ""
echo "=== vs full ggs_baseline ==="
"$PYTHON" "$SCRIPT_DIR/summarize_m1_ablation.py" \
  --metrics_dir "$METRIC_ROOT" --tag_prefix "ggs_ncc_"

echo ""
echo "=== Pass check: corr vs core (need mean overall Δ <= -5e-4, s2d Δ <= 0) ==="
"$PYTHON" - <<'PY'
import json
from pathlib import Path
met = Path("./output_compare/metrics")
rows = []
for p in sorted(met.glob("*_ncc_core_vs_ggs_ncc_corr.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    delta = d.get("delta_optimized_minus_baseline") or {}
    scan = p.name.split("_ncc_core_vs_")[0]
    rows.append((scan, delta))
    print(f"{scan}: overall={delta.get('overall')} s2d={delta.get('mean_s2d')} d2s={delta.get('mean_d2s')}")
if not rows:
    print("No head-to-head JSONs yet.")
else:
    def mean(k):
        vals = [r[1].get(k) for r in rows if isinstance(r[1].get(k), (int, float))]
        return sum(vals) / len(vals) if vals else None
    mo, ms = mean("overall"), mean("mean_s2d")
    print(f"AVG: overall={mo} s2d={ms}")
    ok = mo is not None and ms is not None and mo <= -5e-4 and ms <= 0.0
    print("PASS" if ok else "FAIL")
PY
echo "Done."
