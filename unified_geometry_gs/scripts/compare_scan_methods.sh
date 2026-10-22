#!/usr/bin/env bash
# Compare multiple trained folders against the same baseline on one DTU scan.
#
# Usage:
#   ./scripts/compare_scan_methods.sh <scan_name> <baseline_subdir> <tag>:<opt_subdir> ...
#
# Example (paths relative to unified_geometry_gs):
#   ./scripts/compare_scan_methods.sh scan105 ggs_baseline \
#     "M1-3:m1_ablate_03_full" \
#     "M1-router:m1_router" \
#     "M1-thin:m1_thin"
#
# Writes JSONs to ./output_compare/metrics/${scan}_baseline_vs_<tag>.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

SCAN="$1"
BASE_SUB="$2"
shift 2

BASE="./output_compare/$BASE_SUB/$SCAN"
EVAL="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
MET="./output_compare/metrics"
mkdir -p "$MET"

PYTHON="${PYTHON:-python}"

if [ ! -d "$BASE" ]; then
  echo "ERROR: baseline missing: $BASE"
  exit 1
fi

for entry in "$@"; do
  tag="${entry%%:*}"
  sub="${entry#*:}"
  opt="./output_compare/$sub/$SCAN"
  out="$MET/${SCAN}_baseline_vs_${tag}.json"
  if [ ! -d "$opt" ]; then
    echo "SKIP (missing): $opt"
    continue
  fi
  echo "===== $tag <=> baseline ====="
  "$PYTHON" scripts/compare_models.py \
    --baseline "$BASE" \
    --optimized "$opt" \
    --dtu_eval_root "$EVAL" \
    --out "$out"
done

echo "Done. Metrics in $MET"
