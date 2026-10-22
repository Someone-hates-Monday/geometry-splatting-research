#!/usr/bin/env bash
# M1 — mesh-focused preset (optimize DTU mesh metrics: mean_d2s / mean_s2d / overall).
# Stronger depth–occupancy coupling + earlier regularization than default M1.
#
# Usage:
#   ./scripts/run_m1_mesh_focus.sh <path_to_dtu> [output_dir]
# Example:
#   ./scripts/run_m1_mesh_focus.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1_meshfocus/scan105
#
# Override any knob by exporting before the command, e.g.:
#   LAMBDA_DEPTH_OCC=0.12 ./scripts/run_m1_mesh_focus.sh ...
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults tuned for mesh (may trade a small amount of PSNR/SSIM — check compare_models.py)
export REG_FROM_ITER="${REG_FROM_ITER:-4500}"
export LAMBDA_OCC="${LAMBDA_OCC:-0.06}"
export LAMBDA_DEPTH_OCC="${LAMBDA_DEPTH_OCC:-0.15}"
export OCC_ENTROPY_WEIGHT="${OCC_ENTROPY_WEIGHT:-1.0}"
export OCC_SPARSITY_WEIGHT="${OCC_SPARSITY_WEIGHT:-0.08}"
export DEPTH_OCC_MIN_ALPHA="${DEPTH_OCC_MIN_ALPHA:-0.04}"

exec "$SCRIPT_DIR/run_m1.sh" "$@"
