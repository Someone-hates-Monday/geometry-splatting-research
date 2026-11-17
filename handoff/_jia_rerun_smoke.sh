#!/usr/bin/env bash
set -eo pipefail
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
source configs/dtu_path.env
BASE=output_compare/ggs_trust_rebuild_20261116
SMOKE="$BASE/scan24/smoke"
mkdir -p "$SMOKE" "$BASE/scan24/logs"
python scripts/smoke_jia_trust_fuse_scan24.py \
  -m "$BASE/scan24/model" \
  -s "$UGGS_DTU_ROOT/scan24" \
  -r 2 \
  --iteration 30000 \
  --out_dir "$SMOKE" \
  2>&1 | tee "$BASE/scan24/logs/smoke.log"
echo '=== receipt ==='
cat "$SMOKE/smoke_receipt.json"
ls -la "$SMOKE"
