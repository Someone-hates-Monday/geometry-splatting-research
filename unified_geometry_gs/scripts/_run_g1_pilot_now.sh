#!/usr/bin/env bash
set -eo pipefail
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs

echo "=== available overalls ==="
python - <<'PY'
import json
from pathlib import Path
root = Path("output_compare")
rows = {}
for tau, path in [
    (0.3, "ggs_tau_0.3"),
    (0.5, "ggs_tau_0.5"),
    (0.9, "ggs_ex_trunc4"),
    (1.2, "ggs_tau_1.2"),
]:
    for scan in ("scan24", "scan122"):
        p = root / path / scan / "vis" / "results.json"
        key = f"{scan}@τ={tau}"
        if p.exists():
            d = json.loads(p.read_text())
            rows[key] = d
            print(f"{key}: overall={d['overall']:.6f} d2s={d['mean_d2s']:.6f} s2d={d['mean_s2d']:.6f}")
        else:
            print(f"{key}: MISSING")
PY

echo
echo "=== G1-pilot ==="
python scripts/judge_g1_pilot.py \
  --scans scan24 scan122 \
  --taus 0.3 0.5 0.9 1.2 \
  --out ./output_compare/metrics/g1_pilot_verdict.json

echo
cat ./output_compare/metrics/g1_pilot_verdict.json
