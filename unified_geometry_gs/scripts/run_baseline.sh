#!/usr/bin/env bash
# M0: Run G-GS baseline (no unified occ, no weighting)
# Usage (from unified_geometry_gs): ./scripts/run_baseline.sh <path_to_dtu> <output_dir>
#   After download_dtu.sh, path_to_dtu can be: $UGGS_DTU_ROOT/scan105
#   Or run: source configs/dtu_path.env && ./scripts/run_baseline.sh $UGGS_DTU_ROOT/scan105 ./output/dtu/scan105
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"
python train.py -s "$1" -m "$2" -r 2 --use_decoupled_appearance 3
python mesh_extract.py -m "$2"
