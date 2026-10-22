#!/usr/bin/env bash
# RESEARCH_PROTOCOL P5: full DTU with Pipeline* = no_dn (τ=0.9) + Extract(trunc4).
# Pilot scans 24/105/122 reuse ggs_ex_trunc4 meshes (no retrain).
# Other scans train fresh into ggs_pipeline/<scan>, extract trunc4, compare vs baseline when present.
# Does NOT re-select τ or t — only checks sign of Δoverall vs G-GS baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"

PYTHON="${PYTHON:-python}"
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
BASELINE_ROOT="${BASELINE_ROOT:-./output_compare/ggs_baseline}"
OUT_ROOT="${OUT_ROOT:-./output_compare/ggs_pipeline}"
PILOT_ROOT="${PILOT_ROOT:-./output_compare/ggs_ex_trunc4}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
# Sparse checkpoints — final only (disk)
CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-30000}"

if [ -z "${UGGS_DTU_ROOT:-}" ]; then
  echo "ERROR: UGGS_DTU_ROOT unset"
  exit 1
fi

mkdir -p "$METRIC_ROOT" "$OUT_ROOT"

mapfile -t ALL_SCANS < <(find "$UGGS_DTU_ROOT" -maxdepth 1 -type d -name 'scan*' -printf '%f\n' | sort -V)
if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=("${ALL_SCANS[@]}")
fi

echo "[P5] Pipeline*=no_dn+trunc4 scans=${SCANS[*]}"

is_pilot() {
  case "$1" in
    scan24|scan105|scan122) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_model_dir() {
  local scan="$1"
  if is_pilot "$scan" && [ -f "$PILOT_ROOT/$scan/vis/results.json" ]; then
    echo "$PILOT_ROOT/$scan"
  else
    echo "$OUT_ROOT/$scan"
  fi
}

train_one() {
  local scan="$1"
  local out="$OUT_ROOT/$scan"
  if [ -f "$out/vis/results.json" ]; then
    echo "SKIP train (has results): $out"
    return 0
  fi
  echo "----- P5 train $scan -> $out -----"
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
    --ncc_mask_thresh 0.9
  local rc=$?
  set -e
  if [ "$rc" -eq 2 ]; then
    echo "[P5] soft-stop at $out"
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
  # slim immediately
  rm -f "$out"/chkpnt*.pth
  rm -rf "$out/debug"
}

for scan in "${SCANS[@]}"; do
  if is_pilot "$scan" && [ -f "$PILOT_ROOT/$scan/vis/results.json" ]; then
    echo "SKIP pilot reuse: $PILOT_ROOT/$scan"
    continue
  fi
  train_one "$scan"
done

echo "========== P5 compare vs baseline + sign summary =========="
"$PYTHON" scripts/judge_p5.py \
  --scans "${SCANS[@]}" \
  --pipeline_root "$OUT_ROOT" \
  --pilot_root "$PILOT_ROOT" \
  --baseline_root "$BASELINE_ROOT" \
  --out "$METRIC_ROOT/p5_fulldtu_verdict.json"

echo "=== P5 finished ==="
cat "$METRIC_ROOT/p5_fulldtu_verdict.json"
