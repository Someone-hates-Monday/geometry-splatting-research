#!/usr/bin/env bash
set -eo pipefail
source /home/someone/miniconda3/etc/profile.d/conda.sh
conda activate uggs
cd /mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs

T=/mnt/c/Users/Someone/.cursor/projects/c-Users-Someone-Desktop-SuGaR-main/terminals/59033.txt
echo "=== terminal 59033 ==="
if [ -f "$T" ]; then
  head -n 12 "$T"
  echo "---"
  tail -c 1500 "$T" | tr '\r' '\n' | tail -n 12
else
  echo "missing"
fi

echo
echo "=== train processes ==="
ps aux | grep 'train.py' | grep -v grep || echo none

echo
echo "=== scan105 overalls ==="
python - <<'PY'
import json
from pathlib import Path
root = Path("output_compare")
for tau, tag in [(0.3,"ggs_tau_0.3"),(0.5,"ggs_tau_0.5"),(0.9,"ggs_ex_trunc4"),(1.2,"ggs_tau_1.2")]:
    p = root/tag/"scan105"/"vis"/"results.json"
    if p.exists():
        d=json.loads(p.read_text())
        print(f"τ={tau}: overall={d['overall']:.6f}")
    else:
        out = root/tag/"scan105"
        if (out/"recon_post.ply").exists():
            print(f"τ={tau}: recon_post only")
        elif out.exists():
            pcs=list((out/"point_cloud").glob("iteration_*")) if (out/"point_cloud").exists() else []
            ck=list(out.glob("chkpnt*.pth"))
            print(f"τ={tau}: partial ckpts={len(ck)} iters={[p.name for p in pcs]}")
        else:
            print(f"τ={tau}: MISSING")
PY

echo
if [ -f output_compare/metrics/g1_verdict.json ]; then
  echo "=== g1_verdict exists ==="
  cat output_compare/metrics/g1_verdict.json
else
  echo "no g1_verdict.json yet"
fi
