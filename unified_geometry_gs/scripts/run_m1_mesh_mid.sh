#!/usr/bin/env bash
# M1 — moderate mesh preset (between default M1 and run_m1_mesh_focus.sh).
# Targets better overall Chamfer without the aggressive s2d / PSNR tradeoff of mesh_focus.
#
# Usage:
#   ./scripts/run_m1_mesh_mid.sh <path_to_dtu> [output_dir]
# Example:
#   ./scripts/run_m1_mesh_mid.sh "$UGGS_DTU_ROOT/scan105" ./output_compare/m1_meshmid/scan105
#
# Override: LAMBDA_DEPTH_OCC=0.11 ./scripts/run_m1_mesh_mid.sh ...
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REG_FROM_ITER="${REG_FROM_ITER:-5000}"
export LAMBDA_OCC="${LAMBDA_OCC:-0.04}"
export LAMBDA_DEPTH_OCC="${LAMBDA_DEPTH_OCC:-0.10}"
export OCC_ENTROPY_WEIGHT="${OCC_ENTROPY_WEIGHT:-1.0}"
export OCC_SPARSITY_WEIGHT="${OCC_SPARSITY_WEIGHT:-0.06}"
export DEPTH_OCC_MIN_ALPHA="${DEPTH_OCC_MIN_ALPHA:-0.05}"

exec "$SCRIPT_DIR/run_m1.sh" "$@"
