#!/usr/bin/env bash
# Batch DTU baseline: multiple scans. Requires UGGS_DTU_ROOT (e.g. source configs/dtu_path.env).
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"
for scan in scan24 scan37 scan40 scan55 scan63 scan65 scan69 scan83 scan97 scan105 scan106 scan110 scan114 scan118 scan122; do
  python train.py -s "$UGGS_DTU_ROOT/$scan" -m ./output/dtu/$scan -r 2 --use_decoupled_appearance 3
  python mesh_extract.py -m ./output/dtu/$scan
done
