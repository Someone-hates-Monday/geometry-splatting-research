#!/usr/bin/env bash
# M1: unified occupancy-depth training.
# Usage:
#   ./scripts/run_m1.sh <path_to_dtu> [output_dir]
# Example:
#   ./scripts/run_m1.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1/scan105
#
# Optional env overrides (more optimization attempts without editing this file):
#   LAMBDA_OCC=0.04 LAMBDA_DEPTH_OCC=0.08 REG_FROM_ITER=5000 ./scripts/run_m1.sh ...
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

SCAN_NAME="$(basename "$1")"
OUT_DIR="${2:-./output_compare/m1_unified/${SCAN_NAME}}"

LAMBDA_OCC="${LAMBDA_OCC:-0.02}"
# Default 0.06: multi-scan conservative candidate (docs/EXPERIMENT_LOG_M1.md)
LAMBDA_DEPTH_OCC="${LAMBDA_DEPTH_OCC:-0.06}"
REG_FROM_ITER="${REG_FROM_ITER:-7000}"
OCC_ENTROPY_WEIGHT="${OCC_ENTROPY_WEIGHT:-1.0}"
OCC_SPARSITY_WEIGHT="${OCC_SPARSITY_WEIGHT:-0.05}"
DEPTH_OCC_MIN_ALPHA="${DEPTH_OCC_MIN_ALPHA:-0.05}"

echo "[M1] out=$OUT_DIR lambda_occ=$LAMBDA_OCC lambda_depth_occ=$LAMBDA_DEPTH_OCC reg_from=$REG_FROM_ITER"

python train.py \
  -s "$1" \
  -m "$OUT_DIR" \
  -r 2 \
  --use_decoupled_appearance 3 \
  --lambda_occ "$LAMBDA_OCC" \
  --lambda_depth_occ "$LAMBDA_DEPTH_OCC" \
  --regularization_from_iter "$REG_FROM_ITER" \
  --occ_entropy_weight "$OCC_ENTROPY_WEIGHT" \
  --occ_sparsity_weight "$OCC_SPARSITY_WEIGHT" \
  --depth_occ_min_alpha "$DEPTH_OCC_MIN_ALPHA"
python mesh_extract.py -m "$OUT_DIR"
