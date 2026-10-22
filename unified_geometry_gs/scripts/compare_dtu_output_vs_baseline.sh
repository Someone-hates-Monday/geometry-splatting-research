#!/usr/bin/env bash
# Mesh extract (optional) + DTU mesh metrics: ./output/dtu/<scan> vs ./output/dtu/<scan>_<suffix>.
#
# Usage (from unified_geometry_gs, WSL):
#   source configs/dtu_path.env   # optional; for docs only
#   conda activate uggs
#
#   SCANS="scan65 scan24 scan105 scan122" \
#   SUFFIXES="pm_edge edge_dn pm_edge_go pm_edge_soft" \
#   ./scripts/compare_dtu_output_vs_baseline.sh
#
# Env:
#   SKIP_MESH=1              Skip mesh_extract.py (recon_post.ply already fresh)
#   PYTHON=python            Interpreter
#   DTU_EVAL_ROOT=...        Official DTU SampleSet root (default: ./dtu_eval/Offical_DTU_Dataset)
#   METRICS_DIR=...          Output JSON dir (default: ./output_compare/metrics)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SCANS="${SCANS:-scan65}"
SUFFIXES="${SUFFIXES:-pm_edge edge_dn}"
EVAL="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
MET="${METRICS_DIR:-./output_compare/metrics}"
mkdir -p "$MET"

for scan in $SCANS; do
  base="./output/dtu/$scan"
  if [ ! -d "$base" ]; then
    echo "WARN: skip — missing baseline dir $base"
    continue
  fi
  for suf in $SUFFIXES; do
    opt="./output/dtu/${scan}_${suf}"
    out="$MET/${scan}_baseline_vs_${suf}.json"
    if [ ! -d "$opt" ]; then
      echo "SKIP (missing): $opt"
      continue
    fi
    if [ "${SKIP_MESH:-0}" != "1" ]; then
      echo "===== mesh_extract $opt ====="
      "$PYTHON" mesh_extract.py -m "$opt"
    fi
    echo "===== compare baseline $base vs $opt ====="
    "$PYTHON" scripts/compare_models.py --mesh_only \
      --baseline "$base" \
      --optimized "$opt" \
      --dtu_eval_root "$EVAL" \
      --out "$out"
  done
done

echo "Done. Metrics in $MET"
