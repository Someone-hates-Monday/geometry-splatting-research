#!/usr/bin/env bash
# Visual compare: orbit9 snapshots + stitched rows + DTU vertex error coloring.
#
#   cd unified_geometry_gs && conda activate uggs
#   chmod +x scripts/run_visual_compare_edge.sh
#
#   ./scripts/run_visual_compare_edge.sh
#
# Env (word-split SCANS for multiple):
#   SCANS="scan65 scan24 scan105 scan122"
#   VARIANTS="baseline edge_dn pm_edge"
#   DTU=./dtu_eval/Offical_DTU_Dataset
#   PYTHON=python
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ -f configs/dtu_path.env ]; then
  # shellcheck source=/dev/null
  source configs/dtu_path.env
fi

PYTHON="${PYTHON:-python}"
SCANS="${SCANS:-scan65}"
VARIANTS="${VARIANTS:-baseline edge_dn pm_edge}"
DTU_ROOT="${DTU:-${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}}"

# shellcheck disable=SC2086
exec "$PYTHON" scripts/visual_compare_output_dtu.py \
  --scans $SCANS \
  --variants $VARIANTS \
  --dtu_dir "$DTU_ROOT"
