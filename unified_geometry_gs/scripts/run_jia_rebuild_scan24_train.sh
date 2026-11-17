#!/usr/bin/env bash
# Rebuild protocol: train scan24 once under frozen recipe into ggs_trust_rebuild_20261116.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs

RUN_ROOT="${RUN_ROOT:-./output_compare/ggs_trust_rebuild_20261116/scan24}"
MODEL_DIR="$RUN_ROOT/model"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$MODEL_DIR" "$LOG_DIR"

if [ -z "${UGGS_DTU_ROOT:-}" ]; then
  echo "ERROR: UGGS_DTU_ROOT unset"
  exit 1
fi

# Refuse overwrite of an already-complete model
if [ -f "$MODEL_DIR/point_cloud/iteration_30000/point_cloud.ply" ]; then
  echo "ERROR: model already exists at $MODEL_DIR — refuse overwrite"
  exit 1
fi

{
  echo "=== rebuild train scan24 ==="
  date -u
  echo "git=$(git -C /mnt/c/Users/Someone/Desktop/SuGaR-main rev-parse HEAD 2>/dev/null || true)"
  echo "UGGS_DTU_ROOT=$UGGS_DTU_ROOT"
  echo "MODEL_DIR=$MODEL_DIR"
  nvidia-smi || true
} | tee "$LOG_DIR/train_header.txt"

# Save resolved argv note
cat > "$LOG_DIR/resolved_train_command.txt" <<EOF
python train.py \\
  -s $UGGS_DTU_ROOT/scan24 \\
  -m $MODEL_DIR \\
  -r 2 \\
  --use_decoupled_appearance 3 \\
  --checkpoint_iterations 30000 \\
  --lambda_depth_normal 0 \\
  --lambda_multi_view_ncc 0.6 \\
  --lambda_multi_view_geo 0.02 \\
  --multi_view_ncc_views 1 \\
  --ncc_mask_thresh 0.9
# seed via safe_state: random/np/torch = 0
# densify/other OptimizationParams: current code defaults (not bit-identical claim)
EOF

set +e
python train.py \
  -s "$UGGS_DTU_ROOT/scan24" \
  -m "$MODEL_DIR" \
  -r 2 \
  --use_decoupled_appearance 3 \
  --checkpoint_iterations 30000 \
  --lambda_depth_normal 0 \
  --lambda_multi_view_ncc 0.6 \
  --lambda_multi_view_geo 0.02 \
  --multi_view_ncc_views 1 \
  --ncc_mask_thresh 0.9 \
  2>&1 | tee "$LOG_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e
echo "exit_code=$rc" | tee -a "$LOG_DIR/train_exit.txt"
date -u | tee -a "$LOG_DIR/train_exit.txt"
exit "$rc"
