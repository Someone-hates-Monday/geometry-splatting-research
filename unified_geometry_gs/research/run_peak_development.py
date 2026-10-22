"""Queue isolated peak-aware development checks after the control repeat finishes."""
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
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    names=['research/run_peak_development.py','research/median_peak_adaptive_backend.py',
        'research/audit_peak_backward.py','research/benchmark_peak_forward.py','research/benchmark_peak_step.py',
        'research/median_cuda_backend.py','research/median_adjquad_backend.py',
        'research/median_adjquad_accumulate.inc','research/median_adjquad_refine.inc',
        'submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu',
        'submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu']
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'waiting','after':str(args.after.resolve()),
        'source_sha256':{name:digest(ROOT/name) for name in names},'steps':[],
        'intent':'Development tests only: peak-containing final cells restore subdivision pass five; unchanged high-order adjoint.',
        'limitations':['Selected after a failed development pilot and a one-camera CPU diagnosis; not held-out evidence.',
            'No full training, reconstruction adoption or novelty claim from these checks.']}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    state['source_archive_sha256']=digest(out/'sources.zip');dump(out/'manifest.json',state)
    started=time.monotonic()
    try:
        while True:
            previous=json.loads(args.after.read_text())
            if previous['status']=='complete':break
            if previous['status']=='failed':raise RuntimeError('Control repeat failed; do not start queued GPU work.')
            if time.monotonic()-started>7200:raise TimeoutError('Queue observation limit; do not restart preceding training.')
            time.sleep(10)
        state['status']='running';dump(out/'manifest.json',state)
        def run(stage,command):
            for name,sha in state['source_sha256'].items():
                if digest(ROOT/name)!=sha:raise ValueError(f'Queued source changed: {name}')
            entry={'stage':stage,'status':'running','argv':command};state['steps'].append(entry)
            dump(out/'manifest.json',state);print(f'[{stage}] starting',flush=True)
            start=time.perf_counter()
            with (out/f'{stage}.log').open('w') as log:
                rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
            entry.update({'status':'complete' if rc==0 else 'failed','returncode':rc,'wall_seconds':time.perf_counter()-start})
            dump(out/'manifest.json',state)
            if rc:raise RuntimeError(f'{stage} failed; inspect its preserved log.')
        cpu=['research/results/median_cpu_scan37.json','research/results/median_cpu_scan55.json','research/results/median_cpu_scan63.json']
        backward=out/'backward.json'
        run('backward',[sys.executable,'research/audit_peak_backward.py','--cpu',*cpu,
            '--after','research/results/median_iqi_frozen_v1.json','--out',str(backward)])
        g=json.loads(backward.read_text())
        state['gradient_gate_pass']=g['all_gradients_finite'] and g['all_groups_drift_below_threshold']
        run('forward',[sys.executable,'research/benchmark_peak_forward.py','--cpu',*cpu,
            '--variants','control','adjquad4','peakadj','--out',str(out/'forward.json')])
        if state['gradient_gate_pass']:
            run('step',[sys.executable,'research/benchmark_peak_step.py','--cpu',*cpu,
                '--after',str(backward),'--out',str(out/'step.json')])
        state['status']='complete'
    except BaseException as error:
        state['status']='failed';state['error']=repr(error);raise
    finally:dump(out/'manifest.json',state)


if __name__=='__main__':main()
