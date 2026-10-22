#!/usr/bin/env bash
# M2: geometry confidence weighting (w_geo).
# Usage: ./scripts/run_m2.sh <path_to_dtu> <output_dir>
#
# Optional env overrides:
#   W_GEO_BETA=0.2 W_GEO_MIN=0.8 W_GEO_NCC_ONLY=1 ./scripts/run_m2.sh ...
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

W_GEO_BETA="${W_GEO_BETA:-1.0}"
W_GEO_MIN="${W_GEO_MIN:-0.05}"
W_GEO_NCC_ONLY="${W_GEO_NCC_ONLY:-1}"

python train.py \
  -s "$1" \
  -m "$2" \
  -r 2 \
  --use_decoupled_appearance 3 \
  --use_w_geo \
  --w_geo_beta "$W_GEO_BETA" \
  --w_geo_min "$W_GEO_MIN" \
  $( [ "$W_GEO_NCC_ONLY" = "1" ] && echo "--w_geo_ncc_only" )
python mesh_extract.py -m "$2"
