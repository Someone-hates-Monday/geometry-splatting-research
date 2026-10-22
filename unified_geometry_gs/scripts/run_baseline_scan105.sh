#!/usr/bin/env bash
# Single-scene DTU baseline: scan105. Source dtu_path.env first or set UGGS_DTU_ROOT.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"
python train.py -s "$UGGS_DTU_ROOT/scan105" -m ./output/dtu/scan105 -r 2 --use_decoupled_appearance 3
python mesh_extract.py -m ./output/dtu/scan105
