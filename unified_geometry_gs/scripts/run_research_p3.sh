#!/usr/bin/env bash
# RESEARCH_PROTOCOL P3: τ pilot on scan24 + scan122 (then G1-pilot).
# Train no_dn + ncc_mask_thresh ∈ {0.3,0.5,1.2}; extract locked trunc=4.
# τ=0.9 uses existing ggs_ex_trunc4 (no retrain).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

PYTHON="${PYTHON:-python}"
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-$(seq 10000 1000 30000 | tr '\n' ' ')}"
ABLATION_SKIP_TRAIN="${ABLATION_SKIP_TRAIN:-0}"
SCANS=(scan24 scan122)
TAUS=(0.3 0.5 1.2)

mkdir -p "$METRIC_ROOT"

run_train() {
  local scan="$1" tau="$2"
  local tag
  tag=$(printf "ggs_tau_%s" "$tau")
  local out="./output_compare/${tag}/${scan}"
  echo "----- P3 train $scan τ=$tau -> $out -----"
  # shellcheck disable=SC2086
  set +e
  "$PYTHON" train.py \
    -s "$UGGS_DTU_ROOT/$scan" \
    -m "$out" \
    -r 2 \
    --use_decoupled_appearance 3 \
    --checkpoint_iterations $CHECKPOINT_ITERS \
    --lambda_depth_normal 0 \
    --lambda_multi_view_ncc 0.6 \
    --lambda_multi_view_geo 0.02 \
    --multi_view_ncc_views 1 \
    --ncc_mask_thresh "$tau"
  local rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "[P3] soft-stop at $out"
    exit 0
  elif [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
  "$PYTHON" mesh_extract.py \
    -m "$out" \
    --trunc_voxel_multiplier 4 \
    --voxel_size 0.002 \
    --depth_max 8.0 \
    --num_cluster 1
  "$PYTHON" evaluate_dtu_mesh.py -m "$out" --iteration -1 --DTU "$DTU_EVAL_ROOT"
}

if [ -z "${UGGS_DTU_ROOT:-}" ]; then
  echo "ERROR: UGGS_DTU_ROOT unset"
  exit 1
fi

echo "[P3] scans=${SCANS[*]} taus=${TAUS[*]} extract=trunc4"

if [ "$ABLATION_SKIP_TRAIN" != "1" ]; then
  for tau in "${TAUS[@]}"; do
    for scan in "${SCANS[@]}"; do
      run_train "$scan" "$tau"
    done
  done
fi

# Ensure τ=0.9 absolute overalls (ex_trunc4)
for scan in "${SCANS[@]}"; do
  if [ ! -f "./output_compare/ggs_ex_trunc4/$scan/vis/results.json" ]; then
    "$PYTHON" evaluate_dtu_mesh.py -m "./output_compare/ggs_ex_trunc4/$scan" \
      --iteration -1 --DTU "$DTU_EVAL_ROOT"
  fi
done

echo "========== G1-pilot judgment =========="
"$PYTHON" scripts/judge_g1_pilot.py \
  --scans scan24 scan122 \
  --taus 0.3 0.5 0.9 1.2 \
  --out "$METRIC_ROOT/g1_pilot_verdict.json"

echo "=== P3 + G1-pilot finished ==="
cat "$METRIC_ROOT/g1_pilot_verdict.json"
