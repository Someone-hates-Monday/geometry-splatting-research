#!/usr/bin/env bash
set -e
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs
source configs/dtu_path.env 2>/dev/null || true

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="12.0"

echo "=== mesh availability ==="
for d in ggs_baseline m1_ablate_03_full m1_field_occ; do
  for s in scan24 scan105; do
    if [ -f "./output_compare/$d/$s/recon_post.ply" ]; then
      echo "OK  $d/$s"
    else
      echo "MISSING $d/$s"
    fi
  done
done

EVAL="${DTU_EVAL_ROOT:-./dtu_eval/Offical_DTU_Dataset}"
echo "DTU_EVAL_ROOT=$EVAL"
ls "$EVAL" 2>/dev/null | head -8 || echo "WARN: eval root missing"

chmod +x scripts/compare_scan_methods.sh

for SCAN in scan24 scan105; do
  echo ""
  echo "########## $SCAN ##########"
  ./scripts/compare_scan_methods.sh "$SCAN" ggs_baseline \
    "M1-3:m1_ablate_03_full" \
    "A-field-occ:m1_field_occ"
done

echo ""
echo "=== summarize ==="
python - <<'PY'
import json
from pathlib import Path
met = Path("./output_compare/metrics")
rows = []
for scan in ("scan24", "scan105"):
    for tag in ("M1-3", "A-field-occ"):
        p = met / f"{scan}_baseline_vs_{tag}.json"
        if not p.exists():
            print(f"missing {p}")
            continue
        d = json.loads(p.read_text())
        # flexible key digging
        def dig(obj, *keys):
            for k in keys:
                if isinstance(obj, dict) and k in obj:
                    obj = obj[k]
                else:
                    return None
            return obj
        # print top-level keys once
        print(f"\n{p.name} keys={list(d.keys())}")
        print(json.dumps(d, indent=2)[:2500])
PY
