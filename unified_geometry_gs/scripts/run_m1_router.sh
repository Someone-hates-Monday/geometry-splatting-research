#!/usr/bin/env bash
# M1 + geometry-aware soft router: patch-level routing dynamically scales M1 lambdas.
# See docs/M1_GEOMETRY_ROUTER.md
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

SCAN_NAME="$(basename "$1")"
OUT_DIR="${2:-./output_compare/m1_router/${SCAN_NAME}}"

LAMBDA_OCC="${LAMBDA_OCC:-0.02}"
LAMBDA_DEPTH_OCC="${LAMBDA_DEPTH_OCC:-0.06}"
LAMBDA_DEPTH_OCC_RANK="${LAMBDA_DEPTH_OCC_RANK:-0.0}"
# Set LAMBDA_OCC_THIN=0 to match fixed M1-3-style runs (thin often stacks badly with routing).
LAMBDA_OCC_THIN="${LAMBDA_OCC_THIN:-0.03}"
REG_FROM_ITER="${REG_FROM_ITER:-7000}"
M1_ROUTER_BALANCE="${M1_ROUTER_BALANCE:-0.02}"
M1_ROUTER_LR="${M1_ROUTER_LR:-0.001}"
M1_ROUTER_TEMP="${M1_ROUTER_TEMP:-1.0}"

echo "[M1+router] out=$OUT_DIR balance=$M1_ROUTER_BALANCE temp=$M1_ROUTER_TEMP"

python train.py \
  -s "$1" \
  -m "$OUT_DIR" \
  -r 2 \
  --use_decoupled_appearance 3 \
  --use_m1_geometry_router \
  --m1_router_balance "$M1_ROUTER_BALANCE" \
  --m1_router_lr "$M1_ROUTER_LR" \
  --m1_router_temperature "$M1_ROUTER_TEMP" \
  --lambda_occ "$LAMBDA_OCC" \
  --lambda_depth_occ "$LAMBDA_DEPTH_OCC" \
  --lambda_depth_occ_rank "$LAMBDA_DEPTH_OCC_RANK" \
  --lambda_occ_thin "$LAMBDA_OCC_THIN" \
  --regularization_from_iter "$REG_FROM_ITER" \
  --occ_entropy_weight "${OCC_ENTROPY_WEIGHT:-1.0}" \
  --occ_sparsity_weight "${OCC_SPARSITY_WEIGHT:-0.05}" \
  --depth_occ_min_alpha "${DEPTH_OCC_MIN_ALPHA:-0.05}" \
  --occ_rank_margin "${OCC_RANK_MARGIN:-0.02}" \
  --occ_rank_alpha_soft "${OCC_RANK_ALPHA_SOFT:-0.08}" \
  --occ_thin_edge_power "${OCC_THIN_EDGE_POWER:-1.0}"

python mesh_extract.py -m "$OUT_DIR"
