#!/usr/bin/env bash
# Usage: ./scripts/run_m3.sh <output_dir>
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$UGGS_ROOT"
# python evaluate_dtu_mesh.py -m "$1"
