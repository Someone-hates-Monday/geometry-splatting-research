"""Repeat the baseline suffix from one frozen complete shared-state pilot.

Diagnostic of same-state CUDA training variation, not an independent seed or
permission to replace the pilot quality criteria.
"""
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
    parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    pair=args.pair.resolve()
    base=json.loads((pair/'manifest.json').read_text())
    if base['status']!='complete' or not base['branch_state_verified'] or base['smoke_only']:
        raise ValueError('A verified complete full pilot is required.')
    sources=base['source_sha256'].copy()
    for name,sha in sources.items():
        if digest(ROOT/name)!=sha:raise ValueError(f'Base experiment source changed: {name}')
    sources['research/run_control_repeat.py']=digest(Path(__file__))
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(ROOT/name,name)
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running','steps':[],
        'base_pair':str(pair),'source_sha256':sources,'source_archive_sha256':digest(out/'sources.zip'),
        'seed':0,'iterations':30000,'common_warmup_iteration':6999,'scan':base['scan'],
        'intent':'One baseline repeat from exactly the same common state, diagnosing subsequent CUDA variation.',
        'limitations':['Not an independent random-seed replication or a variance estimate from many runs.',
            'Original candidate quality gates remain unchanged.',
            'Sequential wall clock is not a speed comparison.']}
    dump(out/'manifest.json',state)
    def run(stage,command):
        for name,sha in sources.items():
            if digest(ROOT/name)!=sha:raise ValueError(f'Queued source changed: {name}')
        entry={'stage':stage,'status':'running','argv':command};state['steps'].append(entry)
        dump(out/'manifest.json',state)
        print(f'[{stage}] starting',flush=True)
        started=time.perf_counter()
        with (out/f'{stage}.log').open('w') as log:
            rc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        entry.update({'status':'complete' if rc==0 else 'failed','returncode':rc,'wall_seconds':time.perf_counter()-started})
        dump(out/'manifest.json',state)
        if rc:raise RuntimeError(f'{stage} failed; inspect preserved log.')
    model=out/'control_repeat'
    command=next(s['argv'] for s in base['steps'] if s['stage']=='train_control').copy()
    command[command.index('-m')+1]=str(model)
    try:
        run('train_control_repeat',command)
        integration=json.loads((model/'solver_training_integration.json').read_text())
        if not integration['completed'] or not integration['resume_state_identical'] or integration['initial_state_sha256']!=base['initial_state_sha256']:
            raise ValueError('Repeat did not restore identical common state.')
        if integration['first_camera']!=base['first_camera']:raise ValueError('First camera changed.')
        state['branch_state_verified']=True;state['initial_state_sha256']=integration['initial_state_sha256']
        run('extract',[sys.executable,'mesh_extract.py','-m',str(model),'--iteration','30000',
            '--voxel_size','.002','--trunc_voxel_multiplier','4','--depth_max','8','--num_cluster','1'])
        run('align',[sys.executable,'research/align_solver_mesh.py','--model',str(model)])
        comparisons=[]
        for seed in (0,1,2):
            run(f'evaluate_seed{seed}',[sys.executable,'research/evaluate_mesh_seeded.py',
                '--mesh',str(model/'recon_aligned.ply'),'--scan',base['scan'][4:],'--seed',str(seed),
                '--out',str(out/f'eval_seed{seed}')])
            original=json.loads((pair/f'eval_control_seed{seed}'/'results.json').read_text())
            repeated=json.loads((out/f'eval_seed{seed}'/'results.json').read_text())
            comparisons.append({'seed':seed,'control':original,'control_repeat':repeated,
                'delta_repeat_minus_control':{k:repeated[k]-original[k] for k in ('overall','mean_d2s','mean_s2d')}})
        run('images',[sys.executable,'research/evaluate_images.py','--model',str(model),'--iteration','30000',
            '--split','train','--out',str(out/'images_control_repeat.json')])
        dump(out/'paired_mesh_metrics.json',{'rows':comparisons,
            'mean_delta_repeat_minus_control':{k:sum(r['delta_repeat_minus_control'][k] for r in comparisons)/3
                for k in ('overall','mean_d2s','mean_s2d')},
            'interpretation':'One same-state control retraining difference, not a sampling distribution or significance test.'})
        state['status']='complete'
    except BaseException as error:
        state['status']='failed';state['error']=repr(error);raise
    finally:dump(out/'manifest.json',state)


if __name__=='__main__':main()
