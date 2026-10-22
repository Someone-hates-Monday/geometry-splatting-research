"""Counterbalanced short actual-training timing from the verified common state."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    base=json.loads((args.pair/'manifest.json').read_text())
    audit=json.loads((args.audit/'manifest.json').read_text())
    if base['status']!='complete' or not base['branch_state_verified']:raise ValueError('Verified base required.')
    if audit['status']!='complete' or not audit['gate_pass']:raise ValueError('Full-loss audit gate must pass.')
    sources=base['source_sha256'].copy()
    for name,expected in sources.items():
        if sha(ROOT/name)!=expected:raise ValueError(f'Original source changed: {name}')
    sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__),ROOT/'research/train_sampling_branch.py',ROOT/'research/sampling_batch_backend.py')})
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running','steps':[],
        'source_sha256':sources,'source_archive_sha256':sha(out/'sources.zip'),
        'audit_manifest_sha256':sha(args.audit/'manifest.json'),
        'common_initial_state_sha256':base['initial_state_sha256'],
        'orders':[['control','batch1'],['batch1','control'],['control','batch1']],
        'optimizer_steps_per_run':500,'warmup_steps_per_run':50,
        'scope':'One early scan55 regularized phase; independent process restarts, same checkpoint and seed.',
        'prespecified_engineering_gate':{'geometric_mean_paired_block_wall_ratio_le':.97,'every_paired_block_wall_ratio_le':1.01},
        'limitations':['Protocol chosen after the six frozen-camera development results, before this timing data.',
            'Three process pairs from the same scene/state are timing repetitions, not independent training seeds.',
            'No full-training quality, all-phase throughput or statistical significance claim.']}
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2)+'\n')
    write();rows=[]
    original=next(s['argv'] for s in base['steps'] if s['stage']=='train_control')
    train_args=original[2:].copy()
    if train_args[:2]!=['--solver','control']:raise ValueError('Original wrapper argv changed.')
    train_args=train_args[2:]
    try:
        for i,order in enumerate(state['orders']):
            row={'round':i,'order':order,'variants':{}}
            for v in order:
                for name,expected in sources.items():
                    if sha(ROOT/name)!=expected:raise ValueError(f'Queued source changed: {name}')
                model=out/f'round{i}_{v}'
                argv=[sys.executable,'research/train_sampling_branch.py','--sampling-variant',v,'--timing-steps','500','--timing-warmup','50',*train_args]
                argv[argv.index('-m')+1]=str(model)
                entry={'round':i,'variant':v,'status':'running','argv':argv};state['steps'].append(entry);write()
                started=time.perf_counter()
                with (out/f'round{i}_{v}.log').open('w') as log:
                    rc=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                entry.update({'returncode':rc,'wall_seconds':time.perf_counter()-started,'status':'complete' if rc==0 else 'failed'});write()
                if rc:raise RuntimeError(f'{v} round {i} failed; inspect preserved logs.')
                integration=json.loads((model/'sampling_training_integration.json').read_text())
                if integration['status']!='complete' or integration['initial_state_sha256']!=base['initial_state_sha256'] or integration['first_camera']!=base['first_camera']:
                    raise ValueError('Timing run state mismatch.')
                row['variants'][v]={'block_wall_seconds':integration['measured_block_wall_seconds'],
                    'mean_iteration_gpu_ms':statistics.mean(integration['iteration_gpu_ms']),
                    'median_iteration_gpu_ms':statistics.median(integration['iteration_gpu_ms']),
                    'peak_torch_allocated_bytes':integration['peak_torch_allocated_bytes'],
                    'final_iteration_start_gaussians':integration['gaussian_counts_at_iteration_start'][-1],
                    'backend_sha256':integration['backend_sha256']}
                print(f'round{i}/{v}: {row["variants"][v]}',flush=True)
            row['batch1_control_block_wall_ratio']=row['variants']['batch1']['block_wall_seconds']/row['variants']['control']['block_wall_seconds']
            row['batch1_control_mean_gpu_ratio']=row['variants']['batch1']['mean_iteration_gpu_ms']/row['variants']['control']['mean_iteration_gpu_ms']
            rows.append(row)
        import math
        ratios=[r['batch1_control_block_wall_ratio'] for r in rows]
        geometric=math.exp(statistics.mean(math.log(x) for x in ratios))
        gate=geometric<=.97 and all(x<=1.01 for x in ratios)
        result={'rows':rows,'geometric_mean_paired_block_wall_ratio':geometric,
                'short_phase_timing_gate_pass':gate,'limitations':state['limitations']}
        (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        state['status']='complete';state['gate_pass']=gate
    except BaseException as e:
        state['status']='failed';state['error']=repr(e);raise
    finally:write()


if __name__=='__main__':main()
