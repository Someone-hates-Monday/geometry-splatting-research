#!/usr/bin/env bash
# M1 ablation: train M1-1 .. M1-4 per scan and compare to ggs_baseline.
# Prerequisite: baseline exists at BASELINE_ROOT/<scan> for each scan.
#
# Usage (from unified_geometry_gs):
#   source configs/dtu_path.env
#   export PYTHON=python    # WSL: use conda python, not system python3
#   ./scripts/run_m1_ablation.sh scan24 scan105 scan122
#
# Env:
#   ABLATION_SKIP_TRAIN=1     — only run compare_models + summary (skip train/mesh_extract)
#   BASELINE_ROOT             — default ./output_compare/ggs_baseline
#   DTU_EVAL_ROOT             — default ./dtu_eval/Offical_DTU_Dataset
#   METRIC_ROOT               — default ./output_compare/metrics
#   REG_FROM_ITER             — default 7000
#   LAMBDA_OCC_A1             — default 0.02 (M1-1)
#   LAMBDA_OCC_A3             — default 0.02 (M1-3)
#   LAMBDA_OCC_A4             — default 0.02 (M1-4)
#   LAMBDA_DEPTH_OCC_A2       — default 0.06 (M1-2)
#   LAMBDA_DEPTH_OCC_A3       — default 0.06 (M1-3/4 depth_occ)
#   LAMBDA_DEPTH_OCC_RANK_A4  — default 0.04 (M1-4)
#   PYTHON                    — default python
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UGGS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$UGGS_ROOT/configs/dtu_path.env" ] && source "$UGGS_ROOT/configs/dtu_path.env"
cd "$UGGS_ROOT"
if [ -z "${UGGS_DTU_ROOT:-}" ]; then
  echo "ERROR: UGGS_DTU_ROOT unset. Add configs/dtu_path.env or export UGGS_DTU_ROOT=/path/to/dtu/DTU"
  exit 1
fi

PYTHON="${PYTHON:-python}"
BASELINE_ROOT="${BASELINE_ROOT:-./output_compare/ggs_baseline}"
DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
METRIC_ROOT="${METRIC_ROOT:-./output_compare/metrics}"
REG_FROM_ITER="${REG_FROM_ITER:-7000}"
ABLATION_SKIP_TRAIN="${ABLATION_SKIP_TRAIN:-0}"

LO1="${LAMBDA_OCC_A1:-0.02}"
LO3="${LAMBDA_OCC_A3:-0.02}"
LO4="${LAMBDA_OCC_A4:-0.02}"
LDO2="${LAMBDA_DEPTH_OCC_A2:-0.06}"
LDO3="${LAMBDA_DEPTH_OCC_A3:-0.06}"
LDR4="${LAMBDA_DEPTH_OCC_RANK_A4:-0.04}"

OCC_ENT="${OCC_ENTROPY_WEIGHT:-1.0}"
OCC_SPARSE="${OCC_SPARSITY_WEIGHT:-0.05}"
DEPTH_MINA="${DEPTH_OCC_MIN_ALPHA:-0.05}"
RANK_MARGIN="${OCC_RANK_MARGIN:-0.02}"
RANK_SOFT="${OCC_RANK_ALPHA_SOFT:-0.08}"

if [ "$#" -gt 0 ]; then
  SCANS=("$@")
else
  SCANS=(scan24 scan105 scan122)
fi

mkdir -p "$METRIC_ROOT"

# args: subdir lambda_occ lambda_depth_occ lambda_depth_occ_rank
run_train_one() {
  local sub="$1" lo="$2" ldo="$3" ldr="$4"
  local out="./output_compare/$sub/$scan"
  echo "----- train $scan -> $out (lambda_occ=$lo lambda_depth_occ=$ldo lambda_depth_occ_rank=$ldr) -----"
  "$PYTHON" train.py -s "$UGGS_DTU_ROOT/$scan" -m "$out" -r 2 --use_decoupled_appearance 3 \
    --lambda_occ "$lo" \
    --lambda_depth_occ "$ldo" \
    --lambda_depth_occ_rank "$ldr" \
    --regularization_from_iter "$REG_FROM_ITER" \
    --occ_entropy_weight "$OCC_ENT" \
    --occ_sparsity_weight "$OCC_SPARSE" \
    --depth_occ_min_alpha "$DEPTH_MINA" \
    --occ_rank_margin "$RANK_MARGIN" \
    --occ_rank_alpha_soft "$RANK_SOFT"
  "$PYTHON" mesh_extract.py -m "$out"
}

compare_one() {
  local sub="$1"
  local opt="./output_compare/$sub/$scan"
  local json="$METRIC_ROOT/${scan}_baseline_vs_${sub}.json"
  echo "----- compare $scan -> $json -----"
  "$PYTHON" scripts/compare_models.py \
    --baseline "$BASELINE_ROOT/$scan" \
    --optimized "$opt" \
    --dtu_eval_root "$DTU_EVAL_ROOT" \
    --out "$json"
}

for scan in "${SCANS[@]}"; do
  if [ ! -d "$BASELINE_ROOT/$scan" ]; then
    echo "ERROR: missing baseline: $BASELINE_ROOT/$scan"
    exit 1
  fi
done

if [ "$ABLATION_SKIP_TRAIN" != "1" ]; then
  for scan in "${SCANS[@]}"; do
    echo ""
    echo "========== ABLATION M1-1 occ only: $scan =========="
    run_train_one "m1_ablate_01_occ_only" "$LO1" "0" "0"
  done
  for scan in "${SCANS[@]}"; do
    echo ""
    echo "========== ABLATION M1-2 depth_occ only: $scan =========="
    run_train_one "m1_ablate_02_depth_occ_only" "0" "$LDO2" "0"
  done
  for scan in "${SCANS[@]}"; do
    echo ""
    echo "========== ABLATION M1-3 full: $scan =========="
    run_train_one "m1_ablate_03_full" "$LO3" "$LDO3" "0"
  done
  for scan in "${SCANS[@]}"; do
    echo ""
    echo "========== ABLATION M1-4 full+rank: $scan =========="
    run_train_one "m1_ablate_04_full_rank" "$LO4" "$LDO3" "$LDR4"
  done
fi

for scan in "${SCANS[@]}"; do
  compare_one "m1_ablate_01_occ_only"
  compare_one "m1_ablate_02_depth_occ_only"
  compare_one "m1_ablate_03_full"
  compare_one "m1_ablate_04_full_rank"
done

echo ""
echo "=== Summary (mean delta per ablation tag) ==="
"$PYTHON" "$SCRIPT_DIR/summarize_m1_ablation.py" --metrics_dir "$METRIC_ROOT"
echo "Done."
