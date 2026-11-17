#!/usr/bin/env bash
# Train one rebuild scene under frozen recipe. Usage: run_jia_rebuild_train_one.sh scan105
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

SCAN="${1:?usage: $0 scanNNN}"
case "$SCAN" in
  scan24|scan105|scan122) ;;
  *) echo "ERROR: only scan24/105/122 authorized"; exit 1 ;;
esac

source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs

RUN_ROOT="${RUN_ROOT_BASE:-./output_compare/ggs_trust_rebuild_20261116}/$SCAN"
MODEL_DIR="$RUN_ROOT/model"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$MODEL_DIR" "$LOG_DIR"

if [ -z "${UGGS_DTU_ROOT:-}" ]; then
  echo "ERROR: UGGS_DTU_ROOT unset"
  exit 1
fi
if [ ! -d "$UGGS_DTU_ROOT/$SCAN" ]; then
  echo "ERROR: missing data $UGGS_DTU_ROOT/$SCAN"
  exit 1
fi
if [ -f "$MODEL_DIR/point_cloud/iteration_30000/point_cloud.ply" ]; then
  echo "ERROR: model already exists at $MODEL_DIR — refuse overwrite"
  exit 1
fi

{
  echo "=== rebuild train $SCAN ==="
  date -u
  echo "git=$(git -C /mnt/c/Users/Someone/Desktop/SuGaR-main rev-parse HEAD 2>/dev/null || true)"
  echo "UGGS_DTU_ROOT=$UGGS_DTU_ROOT"
  echo "MODEL_DIR=$MODEL_DIR"
} | tee "$LOG_DIR/train_header.txt"

cat > "$LOG_DIR/resolved_train_command.txt" <<EOF
python train.py -s $UGGS_DTU_ROOT/$SCAN -m $MODEL_DIR -r 2 \\
  --use_decoupled_appearance 3 --checkpoint_iterations 30000 \\
  --lambda_depth_normal 0 --lambda_multi_view_ncc 0.6 --lambda_multi_view_geo 0.02 \\
  --multi_view_ncc_views 1 --ncc_mask_thresh 0.9
EOF

set +e
python train.py \
  -s "$UGGS_DTU_ROOT/$SCAN" \
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
