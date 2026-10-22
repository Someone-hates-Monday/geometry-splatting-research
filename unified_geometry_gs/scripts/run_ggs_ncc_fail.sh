#!/usr/bin/env bash
# NCC fail coverage on top of best known recipe (no depth-normal).
#
# Config = ggs_ablate_no_dn + lambda_ncc_fail (default 0.15):
#   lambda_depth_normal=0, ncc=0.6, geo=0.02
#
# Pass vs ./output_compare/ggs_ablate_no_dn/<scan> (scan24/105/122):
#   mean overall Δ <= -5e-4
#   mean s2d Δ <= 0
#   mean PSNR Δ >= -0.2
#
# Usage:
#   ./scripts/run_ggs_ncc_fail.sh scan24 scan105 scan122
# Soft-stop:
#   ./scripts/request_train_stop.sh ./output_compare/ggs_ncc_fail/scan24
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
NO_DN_ROOT="${NO_DN_ROOT:-./output_compare/ggs_ablate_no_dn}"
BASELINE_ROOT="${BASELINE_ROOT:-./output_compare/ggs_baseline}"
OUT_TAG="${OUT_TAG:-ggs_ncc_fail}"
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
ABLATION_SKIP_TRAIN="${ABLATION_SKIP_TRAIN:-0}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-$(seq 10000 1000 30000 | tr '\n' ' ')}"
LAMBDA_NCC_FAIL="${LAMBDA_NCC_FAIL:-0.15}"
LAMBDA_NCC="${LAMBDA_NCC:-0.6}"
LAMBDA_GEO="${LAMBDA_GEO:-0.02}"

if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=(scan24 scan105 scan122)
fi

mkdir -p "$METRIC_ROOT"
chmod +x "$SCRIPT_DIR/request_train_stop.sh" 2>/dev/null || true

run_train_one() {
  local scan="$1"
  local out="./output_compare/$OUT_TAG/$scan"
  echo "----- train $scan -> $out (ncc-fail=$LAMBDA_NCC_FAIL) -----"
  # shellcheck disable=SC2086
  set +e
  "$PYTHON" train.py \
    -s "$UGGS_DTU_ROOT/$scan" \
    -m "$out" \
    -r 2 \
    --use_decoupled_appearance 3 \
    --checkpoint_iterations $CHECKPOINT_ITERS \
    --lambda_depth_normal 0 \
    --lambda_multi_view_ncc "$LAMBDA_NCC" \
    --lambda_multi_view_geo "$LAMBDA_GEO" \
    --lambda_ncc_fail "$LAMBDA_NCC_FAIL" \
    ${START_CHECKPOINT:+--start_checkpoint "$START_CHECKPOINT"}
  local rc=$?
  set -e
  START_CHECKPOINT=""
  if [ "$rc" -eq 2 ]; then
    echo "[ncc-fail] soft-stop at $out"
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
  if [ ! -f "$opt/recon_post.ply" ] || [ ! -d "$base" ]; then
    echo "SKIP compare $tag $scan"
    return 0
  fi
  echo "----- compare $tag $scan -----"
  "$PYTHON" scripts/compare_models.py \
    --baseline "$base" --optimized "$opt" \
    --dtu_eval_root "$DTU_EVAL_ROOT" --out "$json"
}

for scan in "${SCANS[@]}"; do
  [ -d "$NO_DN_ROOT/$scan" ] || { echo "ERROR: missing control $NO_DN_ROOT/$scan"; exit 1; }
done

echo "[ncc-fail] scans=${SCANS[*]} lambda_ncc_fail=$LAMBDA_NCC_FAIL ncc=$LAMBDA_NCC geo=$LAMBDA_GEO"

if [ "$ABLATION_SKIP_TRAIN" != "1" ]; then
  for scan in "${SCANS[@]}"; do
    echo ""
    echo "========== NCC-fail cover — $scan =========="
    run_train_one "$scan"
  done
fi

for scan in "${SCANS[@]}"; do
  compare_pair "$NO_DN_ROOT/$scan" "./output_compare/$OUT_TAG/$scan" \
    "no_dn_vs_${OUT_TAG}" "$scan"
  compare_pair "$BASELINE_ROOT/$scan" "./output_compare/$OUT_TAG/$scan" \
    "baseline_vs_${OUT_TAG}" "$scan"
done

echo ""
echo "=== Verdict vs no_dn (mesh: overall<=-5e-4 & s2d<=0; image: psnr>=-0.2) ==="
OUT_TAG="$OUT_TAG" "$PYTHON" - <<'PY'
import json, os
from pathlib import Path
tag = os.environ.get("OUT_TAG", "ggs_ncc_fail")
met = Path("./output_compare/metrics")
rows = []
print(f"head-to-head: {tag} − no_dn\n")
for scan in ("scan24", "scan105", "scan122"):
    p = met / f"{scan}_no_dn_vs_{tag}.json"
    if not p.exists():
        print(f"  {scan}: MISSING")
        continue
    d = json.loads(p.read_text(encoding="utf-8")).get("delta_optimized_minus_baseline") or {}
    rows.append(d)
    print(
        f"  {scan}: overall={d.get('overall'):+.6e} "
        f"s2d={d.get('mean_s2d'):+.6e} psnr={d.get('psnr'):+.4f}"
    )
if rows:
    def mean(k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return sum(v) / len(v) if v else None
    mo, ms, mp = mean("overall"), mean("mean_s2d"), mean("psnr")
    print(f"  AVG(n={len(rows)}): overall={mo:+.6e} s2d={ms:+.6e} psnr={mp:+.4f}")
    ok = (
        mo is not None and ms is not None and mp is not None
        and mo <= -5e-4 and ms <= 0.0 and mp >= -0.2
    )
    print(
        "VERDICT:",
        "PASS (ncc-fail is next main recipe)" if ok else "FAIL (keep no_dn as main recipe)",
    )
PY
echo "Done."
