#!/usr/bin/env bash
# Route B: M1-3 + normal-guided occupancy sharpening (mesh-aligned surrogate).
# Train + mesh_extract per scan; compare to baseline when BASELINE_ROOT is set.
#
# Usage (from unified_geometry_gs):
#   source configs/dtu_path.env
#   export PYTHON=python
#   ./scripts/run_m1_b_mesh_align.sh scan24 scan105 scan122
#
# Env:
#   LAMBDA_OCC_NG_SHARP — default 0.015 (tune 0.01–0.03)
#   OCC_NG_SHARP_BETA   — default 5.0
#   REG_FROM_ITER       — default 7000
#   BASELINE_ROOT       — default ./output_compare/ggs_baseline
#   SKIP_TRAIN=1        — only compare_models
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
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
REG_FROM_ITER="${REG_FROM_ITER:-7000}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

LO="${LAMBDA_OCC_A3:-0.02}"
LDO="${LAMBDA_DEPTH_OCC_A3:-0.06}"
LNG="${LAMBDA_OCC_NG_SHARP:-0.015}"
BETA="${OCC_NG_SHARP_BETA:-5.0}"

OCC_ENT="${OCC_ENTROPY_WEIGHT:-1.0}"
OCC_SPARSE="${OCC_SPARSITY_WEIGHT:-0.05}"
DEPTH_MINA="${DEPTH_OCC_MIN_ALPHA:-0.05}"

SUBDIR="${M1_B_SUBDIR:-m1_b_mesh_align}"

if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=(scan24 scan105 scan122)
fi

mkdir -p "$METRIC_ROOT"

for scan in "${SCANS[@]}"; do
  if [ ! -d "$BASELINE_ROOT/$scan" ]; then
    echo "WARN: missing baseline $BASELINE_ROOT/$scan (compare will fail for this scan)"
  fi
done

train_one() {
  local scan="$1"
  local out="./output_compare/$SUBDIR/$scan"
  echo "----- M1-B train $scan -> $out (lambda_occ_ng_sharp=$LNG beta=$BETA) -----"
  "$PYTHON" train.py -s "$UGGS_DTU_ROOT/$scan" -m "$out" -r 2 --use_decoupled_appearance 3 \
    --lambda_occ "$LO" \
    --lambda_depth_occ "$LDO" \
    --lambda_depth_occ_rank 0 \
    --lambda_occ_ng_sharp "$LNG" \
    --occ_ng_sharp_beta "$BETA" \
    --regularization_from_iter "$REG_FROM_ITER" \
    --occ_entropy_weight "$OCC_ENT" \
    --occ_sparsity_weight "$OCC_SPARSE" \
    --depth_occ_min_alpha "$DEPTH_MINA"
  "$PYTHON" mesh_extract.py -m "$out"
}

compare_one() {
  local scan="$1"
  local opt="./output_compare/$SUBDIR/$scan"
  local json="$METRIC_ROOT/${scan}_baseline_vs_${SUBDIR}.json"
  echo "----- compare $scan -> $json -----"
  "$PYTHON" scripts/compare_models.py \
    --baseline "$BASELINE_ROOT/$scan" \
    --optimized "$opt" \
    --dtu_eval_root "$DTU_EVAL_ROOT" \
    --out "$json"
}

if [ "$SKIP_TRAIN" != "1" ]; then
  for scan in "${SCANS[@]}"; do
    train_one "$scan"
  done
fi

for scan in "${SCANS[@]}"; do
  if [ -d "$BASELINE_ROOT/$scan" ]; then
    compare_one "$scan"
  fi
done

echo "Done. Metrics under $METRIC_ROOT ; models under ./output_compare/$SUBDIR/"
