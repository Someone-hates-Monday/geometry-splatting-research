#!/usr/bin/env bash
# M1 + thin-boundary helper:
#   L_occ + L_depth_occ + optional L_occ_rank + L_occ_thin
# Goal: improve thin walls / concave boundaries by suppressing ambiguous alpha bands.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

SCAN_NAME="$(basename "$1")"
OUT_DIR="${2:-./output_compare/m1_thin/${SCAN_NAME}}"

LAMBDA_OCC="${LAMBDA_OCC:-0.02}"
LAMBDA_DEPTH_OCC="${LAMBDA_DEPTH_OCC:-0.06}"
LAMBDA_DEPTH_OCC_RANK="${LAMBDA_DEPTH_OCC_RANK:-0.00}"
LAMBDA_OCC_THIN="${LAMBDA_OCC_THIN:-0.03}"
OCC_THIN_EDGE_POWER="${OCC_THIN_EDGE_POWER:-1.0}"
REG_FROM_ITER="${REG_FROM_ITER:-7000}"
OCC_RANK_MARGIN="${OCC_RANK_MARGIN:-0.02}"
OCC_RANK_ALPHA_SOFT="${OCC_RANK_ALPHA_SOFT:-0.08}"
OCC_ENTROPY_WEIGHT="${OCC_ENTROPY_WEIGHT:-1.0}"
OCC_SPARSITY_WEIGHT="${OCC_SPARSITY_WEIGHT:-0.05}"
DEPTH_OCC_MIN_ALPHA="${DEPTH_OCC_MIN_ALPHA:-0.05}"

echo "[M1+thin] out=$OUT_DIR lambda_occ=$LAMBDA_OCC lambda_depth_occ=$LAMBDA_DEPTH_OCC lambda_rank=$LAMBDA_DEPTH_OCC_RANK lambda_thin=$LAMBDA_OCC_THIN"

python train.py \
  -s "$1" \
  -m "$OUT_DIR" \
  -r 2 \
  --use_decoupled_appearance 3 \
  --lambda_occ "$LAMBDA_OCC" \
  --lambda_depth_occ "$LAMBDA_DEPTH_OCC" \
  --lambda_depth_occ_rank "$LAMBDA_DEPTH_OCC_RANK" \
  --lambda_occ_thin "$LAMBDA_OCC_THIN" \
  --occ_thin_edge_power "$OCC_THIN_EDGE_POWER" \
  --regularization_from_iter "$REG_FROM_ITER" \
  --occ_rank_margin "$OCC_RANK_MARGIN" \
  --occ_rank_alpha_soft "$OCC_RANK_ALPHA_SOFT" \
  --occ_entropy_weight "$OCC_ENTROPY_WEIGHT" \
  --occ_sparsity_weight "$OCC_SPARSITY_WEIGHT" \
  --depth_occ_min_alpha "$DEPTH_OCC_MIN_ALPHA"
python mesh_extract.py -m "$OUT_DIR"
