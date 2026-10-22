#!/usr/bin/env bash
# Run M1 (unified occupancy-depth) into a dedicated compare folder.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

OUT_DIR="./output_compare/m1_unified/scan105"
python train.py \
  -s "$UGGS_DTU_ROOT/scan105" \
  -m "$OUT_DIR" \
  -r 2 \
  --use_decoupled_appearance 3 \
  --lambda_occ 0.02 \
  --lambda_depth_occ 0.05
python mesh_extract.py -m "$OUT_DIR"
