#!/usr/bin/env bash
# G-GS leave-one-out / multi-view split ablations vs full ggs_baseline.
#
# Full baseline must exist: ./output_compare/ggs_baseline/<scan>
#
# Variants (select with GGS_VARIANTS, comma-separated; default = all four):
#   no_dn     — lambda_depth_normal=0
#   no_mv     — lambda_multi_view_ncc=0 lambda_multi_view_geo=0
#   ncc_only  — lambda_multi_view_geo=0  (keep NCC=0.6)     [phase-1 MV split]
#   geo_only  — lambda_multi_view_ncc=0  (keep geo=0.02)    [phase-1 MV split]
#
# Usage:
#   ./scripts/run_ggs_ablation.sh scan24 scan105
#   GGS_VARIANTS=ncc_only,geo_only ./scripts/run_ggs_ablation.sh scan24 scan105
#
# Soft-stop:
#   ./scripts/request_train_stop.sh ./output_compare/ggs_ablate_ncc_only/scan24
#
# Env: ABLATION_SKIP_TRAIN=1 | START_CHECKPOINT=... | CHECKPOINT_ITERS=... | PYTHON=python
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

if [ -z "${UGGS_DTU_ROOT:-}" ]; then
  echo "ERROR: UGGS_DTU_ROOT unset. source configs/dtu_path.env first."
  exit 1
fi

PYTHON="${PYTHON:-python}"
BASELINE_ROOT="${BASELINE_ROOT:-./output_compare/ggs_baseline}"
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
ABLATION_SKIP_TRAIN="${ABLATION_SKIP_TRAIN:-0}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-$(seq 10000 1000 30000 | tr '\n' ' ')}"
GGS_VARIANTS="${GGS_VARIANTS:-no_dn,no_mv,ncc_only,geo_only}"

if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=(scan24 scan105)
fi

mkdir -p "$METRIC_ROOT"
chmod +x "$SCRIPT_DIR/request_train_stop.sh" 2>/dev/null || true

IFS=',' read -r -a VARIANT_LIST <<< "$GGS_VARIANTS"

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
    "$@" \
    ${START_CHECKPOINT:+--start_checkpoint "$START_CHECKPOINT"}
  local rc=$?
  set -e
  START_CHECKPOINT=""
  if [ "$rc" -eq 2 ]; then
    echo "[ggs-ablate] soft-stop at $out — skip mesh_extract; re-run later to finish."
    exit 0
  elif [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
  "$PYTHON" mesh_extract.py -m "$out"
}

compare_one() {
  local sub="$1"
  local scan="$2"
  local opt="./output_compare/$sub/$scan"
  local json="$METRIC_ROOT/${scan}_baseline_vs_${sub}.json"
  if [ ! -f "$opt/recon_post.ply" ]; then
    echo "SKIP compare (no recon_post.ply): $opt"
    return 0
  fi
  echo "----- compare $scan $sub -> $json -----"
  "$PYTHON" scripts/compare_models.py \
    --baseline "$BASELINE_ROOT/$scan" \
    --optimized "$opt" \
    --dtu_eval_root "$DTU_EVAL_ROOT" \
    --out "$json"
}

for scan in "${SCANS[@]}"; do
  if [ ! -d "$BASELINE_ROOT/$scan" ]; then
    echo "ERROR: missing baseline: $BASELINE_ROOT/$scan"
    exit 1
  fi
done

echo "[ggs-ablate] variants=$GGS_VARIANTS scans=${SCANS[*]}"

if [ "$ABLATION_SKIP_TRAIN" != "1" ]; then
  if has_variant no_dn; then
    for scan in "${SCANS[@]}"; do
      echo ""
      echo "========== G-GS LOO: no depth-normal — $scan =========="
      run_train_one "ggs_ablate_no_dn" "$scan" --lambda_depth_normal 0
    done
  fi
  if has_variant no_mv; then
    for scan in "${SCANS[@]}"; do
      echo ""
      echo "========== G-GS LOO: no multi-view — $scan =========="
      run_train_one "ggs_ablate_no_mv" "$scan" \
        --lambda_multi_view_ncc 0 --lambda_multi_view_geo 0
    done
  fi
  if has_variant ncc_only; then
    for scan in "${SCANS[@]}"; do
      echo ""
      echo "========== G-GS MV-split: NCC only (geo=0) — $scan =========="
      run_train_one "ggs_ablate_ncc_only" "$scan" --lambda_multi_view_geo 0
    done
  fi
  if has_variant geo_only; then
    for scan in "${SCANS[@]}"; do
      echo ""
      echo "========== G-GS MV-split: geo only (ncc=0) — $scan =========="
      run_train_one "ggs_ablate_geo_only" "$scan" --lambda_multi_view_ncc 0
    done
  fi
fi

for scan in "${SCANS[@]}"; do
  has_variant no_dn && compare_one "ggs_ablate_no_dn" "$scan"
  has_variant no_mv && compare_one "ggs_ablate_no_mv" "$scan"
  has_variant ncc_only && compare_one "ggs_ablate_ncc_only" "$scan"
  has_variant geo_only && compare_one "ggs_ablate_geo_only" "$scan"
done

echo ""
echo "=== Summary (Δ = variant − full ggs_baseline; overall↑ => removed/kept module matters) ==="
"$PYTHON" "$SCRIPT_DIR/summarize_m1_ablation.py" \
  --metrics_dir "$METRIC_ROOT" \
  --tag_prefix "ggs_ablate_"
echo "Done."
