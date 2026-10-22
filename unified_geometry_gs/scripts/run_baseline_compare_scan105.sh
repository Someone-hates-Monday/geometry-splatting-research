#!/usr/bin/env bash
# Run baseline into a dedicated compare folder.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

OUT_DIR="./output_compare/ggs_baseline/scan105"
python train.py -s "$UGGS_DTU_ROOT/scan105" -m "$OUT_DIR" -r 2 --use_decoupled_appearance 3
python mesh_extract.py -m "$OUT_DIR"
