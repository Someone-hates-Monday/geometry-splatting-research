#!/usr/bin/env bash
# RESEARCH_PROTOCOL P4: add scan105 to τ grid {0.3,0.5,1.2}; τ=0.9 = ggs_ex_trunc4.
# Extract locked trunc=4. Then G1 on three scans.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

PYTHON="${PYTHON:-python}"
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-$(seq 10000 1000 30000 | tr '\n' ' ')}"
SCAN=scan105
TAUS=(0.3 0.5 1.2)

if [ -z "${UGGS_DTU_ROOT:-}" ]; then
  echo "ERROR: UGGS_DTU_ROOT unset"
  exit 1
fi

mkdir -p "$METRIC_ROOT"

run_train() {
  local tau="$1"
  local tag
  tag=$(printf "ggs_tau_%s" "$tau")
  local out="./output_compare/${tag}/${SCAN}"
  if [ -f "$out/vis/results.json" ]; then
    echo "SKIP done $out"
    return 0
  fi
  echo "----- P4 train $SCAN τ=$tau -> $out -----"
  # shellcheck disable=SC2086
  set +e
  "$PYTHON" train.py \
    -s "$UGGS_DTU_ROOT/$SCAN" \
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
    echo "[P4] soft-stop at $out"
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

echo "[P4] scan=$SCAN taus=${TAUS[*]} extract=trunc4"

for tau in "${TAUS[@]}"; do
  run_train "$tau"
done

# ensure τ=0.9 ref
if [ ! -f "./output_compare/ggs_ex_trunc4/${SCAN}/vis/results.json" ]; then
  echo "ERROR: missing τ=0.9 mesh metrics at ggs_ex_trunc4/${SCAN}"
  exit 1
fi

echo "========== G1 (three scans) =========="
"$PYTHON" scripts/judge_g1.py \
  --scans scan24 scan105 scan122 \
  --taus 0.3 0.5 0.9 1.2 \
  --out "$METRIC_ROOT/g1_verdict.json"

echo "=== P4 + G1 finished ==="
cat "$METRIC_ROOT/g1_verdict.json"
