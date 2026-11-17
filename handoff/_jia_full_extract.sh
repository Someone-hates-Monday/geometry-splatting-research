#!/usr/bin/env bash
# Conditional P1 pass → full 24→105→122 trust extract (seed=0). No retrain.
set -eo pipefail
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
source configs/dtu_path.env
export DTU_EVAL_ROOT="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"

BASE=output_compare/ggs_trust_rebuild_20261116
STAMP=$(date +%Y%m%d_%H%M%S)
ID_DIR="$BASE/source_identity_${STAMP}"
mkdir -p "$BASE/scan24/logs"

echo "===== snapshot ${STAMP} ====="
python scripts/snapshot_jia_trust_source_identity.py "$ID_DIR" \
  2>&1 | tee "$BASE/scan24/logs/snapshot_${STAMP}.log"

echo "===== selfcheck ====="
python scripts/selfcheck_jia_eval_seed_and_snapshot.py \
  "$ID_DIR/source_identity_manifest.json" \
  "$BASE/scan24/compat_grad_20261117_145006/compat_receipt.json" \
  2>&1 | tee "$BASE/scan24/logs/selfcheck_${STAMP}.log"

echo "===== full extract 24→105→122 ====="
export JIA_SOURCE_IDENTITY_DIR="$ID_DIR"
export JIA_RUN_STAMP="$STAMP"
python scripts/run_jia_trust_full_extract.py \
  2>&1 | tee "$BASE/full_extract_${STAMP}.log"

echo "===== done stamp=${STAMP} ====="
ls -la "$BASE"/verdict_*.json "$BASE"/paired_results_*.csv 2>/dev/null | tail -n 20
