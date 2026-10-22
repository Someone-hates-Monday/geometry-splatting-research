"""Sequential expanded gradient audit and frozen full-loss timing for FP64 VJP."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'research/precise_preprocess_backend.py',ROOT/'research/median_cuda_backend.py',
        ROOT/'research/audit_precise_full_loss.py',ROOT/'research/benchmark_precise_preprocess.py',
        ROOT/'research/benchmark_median_cuda.py',ROOT/'gaussian_renderer/__init__.py',ROOT/'utils/loss_utils.py']
    raster=ROOT/'submodules/diff-gaussian-rasterization'
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        sources.extend(raster.glob(pattern))
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT).as_posix())
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running','steps':[],
        'source_sha256':hashes,'source_archive_sha256':sha(out/'sources.zip'),
        'gradient_gate':.001,
        'intent':'Additional frozen complete-loss gradients against precise2 plus timing versus original and precise2.',
        'limitations':['Exploratory follow-up to failed float32 expanded audit and two selected diagnostic cameras.',
            'FP64 preprocessing is not an independent derivative oracle or proof of end-to-end determinism.',
            'No optimizer, densification or quality training.']}
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    write()
    cpu=[str(ROOT/f'research/results/median_cpu_scan{s}.json') for s in (37,55,63)]
    try:
        for stage,script in (('audit','audit_precise_full_loss.py'),('timing','benchmark_precise_preprocess.py')):
            for name,expected in hashes.items():
                if sha(ROOT/name)!=expected:raise ValueError(f'Queued source changed: {name}')
            argv=[sys.executable,f'research/{script}','--cpu',*cpu,'--out',str(out/stage)]
            entry={'stage':stage,'argv':argv,'status':'running'};state['steps'].append(entry);write()
            started=time.perf_counter()
            with (out/f'{stage}.log').open('w') as log:rc=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            entry.update({'status':'complete' if rc==0 else 'failed','returncode':rc,'wall_seconds':time.perf_counter()-started});write()
            if rc:raise RuntimeError(f'{stage} failed; preserve outputs.')
        audit=json.loads((out/'audit/manifest.json').read_text())
        state['expanded_gradient_gate_pass']=audit['gate_pass'];state['status']='complete'
    except BaseException as e:
        state['status']='failed';state['error']=repr(e);raise
    finally:write()


if __name__=='__main__':main()
