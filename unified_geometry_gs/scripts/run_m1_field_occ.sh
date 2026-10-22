#!/usr/bin/env bash
# A: field-intrinsic occupancy on T=1/2 interface slope.
# Replaces screen L_occ / L_doc — do NOT combine with --lambda_occ / --lambda_depth_occ.
#
# Usage:
#   ./scripts/run_m1_field_occ.sh <path_to_dtu> [output_dir]
# Example:
#   ./scripts/run_m1_field_occ.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1_field_occ/scan105
#
# Env overrides:
#   LAMBDA_FIELD_OCC=0.02 FIELD_OCC_BETA=1.0 REG_FROM_ITER=7000 ./scripts/run_m1_field_occ.sh ...
#   START_CHECKPOINT=./output_compare/m1_field_occ/scan24/chkpnt15000.pth \
#     ./scripts/run_m1_field_occ.sh "$UGGS_DTU_ROOT/scan24" ./output_compare/m1_field_occ/scan24
#
# Soft-stop (save next checkpoint then exit):
#   ./scripts/request_train_stop.sh ./output_compare/m1_field_occ/scan24
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

SCAN_NAME="$(basename "$1")"
OUT_DIR="${2:-./output_compare/m1_field_occ/${SCAN_NAME}}"

LAMBDA_FIELD_OCC="${LAMBDA_FIELD_OCC:-0.02}"
FIELD_OCC_BETA="${FIELD_OCC_BETA:-1.0}"
FIELD_OCC_CLAMP="${FIELD_OCC_CLAMP:-10.0}"
REG_FROM_ITER="${REG_FROM_ITER:-7000}"
# Every 1000 iters from 10k..30k (override with CHECKPOINT_ITERS="15000 20000 25000 30000")
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-$(seq 10000 1000 30000 | tr '\n' ' ')}"

echo "[A field-occ] out=$OUT_DIR lambda_field_occ=$LAMBDA_FIELD_OCC beta=$FIELD_OCC_BETA reg_from=$REG_FROM_ITER"
echo "[A field-occ] checkpoints: $CHECKPOINT_ITERS"
if [ -n "${START_CHECKPOINT:-}" ]; then
  echo "[A field-occ] resume: $START_CHECKPOINT"
fi

# shellcheck disable=SC2086
set +e
python train.py \
  -s "$1" \
  -m "$OUT_DIR" \
  -r 2 \
  --use_decoupled_appearance 3 \
  --lambda_occ 0 \
  --lambda_depth_occ 0 \
  --lambda_field_occ "$LAMBDA_FIELD_OCC" \
  --field_occ_beta "$FIELD_OCC_BETA" \
  --field_occ_clamp_mult "$FIELD_OCC_CLAMP" \
  --regularization_from_iter "$REG_FROM_ITER" \
  --checkpoint_iterations $CHECKPOINT_ITERS \
  ${START_CHECKPOINT:+--start_checkpoint "$START_CHECKPOINT"}
train_rc=$?
set -e
if [ "$train_rc" -eq 2 ]; then
  echo "[A field-occ] soft-stop: skip mesh_extract (resume later from latest chkpnt*.pth)"
  exit 0
elif [ "$train_rc" -ne 0 ]; then
  exit "$train_rc"
fi
python mesh_extract.py -m "$OUT_DIR"
