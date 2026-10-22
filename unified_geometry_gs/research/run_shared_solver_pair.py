"""Shared warmup and complete runtime-state branching for a solver pilot.

Smoke is an integration test with altered regularization timing. Full mode keeps
the 30k recipe, branches after 6999, and never claims training-seed replication.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',default='scan55')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--integration-gate',type=Path)
    args=parser.parse_args()
    os.chdir(ROOT)
    if not args.scan.startswith('scan') or not args.scan[4:].isdigit():
        parser.error('Expected scanNN.')
    source=ROOT.parent/'dtu/DTU'/args.scan
    if not source.is_dir():
        parser.error('DTU source missing.')
    for p in ('adjquad_backward_v1.json','adjquad_math_v1.json','adjquad_step_timing_v1.json','checkpoint_restore_v1.json'):
        if not (ROOT/'research/results'/p).is_file():
            raise ValueError(f'Preflight incomplete: {p}')
    g=json.loads((ROOT/'research/results/adjquad_backward_v1.json').read_text())
    if not g['all_groups_drift_below_threshold'] or not g['all_gradients_finite']:
        raise ValueError('Gradient gate failed.')
    r=json.loads((ROOT/'research/results/checkpoint_restore_v1.json').read_text())
    if not r['cases']['corrected']['parameter_binding_correct']:
        raise ValueError('Restore binding not validated.')
    if not args.smoke:
        if args.integration_gate is None:
            parser.error('Full pilot requires completed smoke manifest.')
        gate=json.loads(args.integration_gate.read_text())
        if gate['status']!='complete' or not gate['smoke_only'] or not gate['branch_state_verified']:
            raise ValueError('Integration smoke not passed.')
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    iterations=32 if args.smoke else 30000
    warmup=12 if args.smoke else 6999
    reg=warmup+1
    import torch
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'scan':args.scan,
        'smoke_only':args.smoke,'iterations':iterations,'warmup_stop':warmup,'regularization_start':reg,
        'status':'running','steps':[],'branch_state_verified':False,
        'order':['control','adjquad4'],'seed':0,
        'selection':'Development pilot chosen after frozen-model diagnostics; not a held-out confirmatory scene.',
        'source_sha256':{},'environment':{'torch':torch.__version__,'python':sys.version,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
        'limitations':['Shared warmup plus state equality eliminates initial warmup mismatch, not subsequent CUDA nondeterminism.',
            'One scene, one training seed, sequential fixed arm order; no statistical or training-speed claim.',
            'Existing eval=False loads training cameras; image evaluation is training-view only.',
            'Both arms use research corrected restore with otherwise unchanged full training recipe.']}
    names=['train.py','arguments/__init__.py','scene/gaussian_model.py','scene/__init__.py',
        'gaussian_renderer/__init__.py','utils/loss_utils.py','utils/general_utils.py',
        'research/run_shared_solver_pair.py','research/train_solver_branch.py','research/checkpoint_state.py',
        'research/median_cuda_backend.py','research/median_adjquad_backend.py',
        'research/median_adjquad_accumulate.inc','research/median_adjquad_refine.inc',
        'mesh_extract.py','evaluate_dtu_mesh.py','dtu_eval/eval.py',
        'research/align_solver_mesh.py','research/evaluate_mesh_seeded.py','research/evaluate_images.py']
    rasterizer=ROOT/'submodules/diff-gaussian-rasterization'
    for pat in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        names.extend(p.relative_to(ROOT).as_posix() for p in rasterizer.glob(pat))
    manifest['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(set(names))}
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in manifest['source_sha256']:archive.write(ROOT/name,name)
    manifest['source_archive_sha256']=hashlib.sha256((out/'sources.zip').read_bytes()).hexdigest()
    dump(out/'manifest.json',manifest)
    def run(stage,command):
        for name,digest in manifest['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Experiment source changed: {name}')
        entry={'stage':stage,'argv':command,'status':'running'}
        manifest['steps'].append(entry);dump(out/'manifest.json',manifest)
        started=time.perf_counter()
        print(f'[{stage}] starting',flush=True)
        with (out/f'{stage}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        entry.update({'wall_seconds':time.perf_counter()-started,'returncode':result.returncode,
            'status':'complete' if result.returncode==0 else 'failed'})
        dump(out/'manifest.json',manifest)
        if result.returncode:raise RuntimeError(f'{stage} failed; inspect preserved log.')
    common=out/'common'
    checkpoint=common/f'chkpnt{warmup}.pth'
    sidecar=checkpoint.with_suffix('.runtime.pth')
    def training(solver,model,extra):
        return [sys.executable,'research/train_solver_branch.py','--solver',solver,
            '-s',str(source),'-m',str(model),'-r','2','--dataset','dtu','--use_decoupled_appearance','3',
            '--iterations',str(iterations),'--regularization_from_iter',str(reg),
            '--save_iterations',str(iterations),'--test_iterations',str(iterations),
            '--checkpoint_iterations',str(warmup if model==common else iterations),
            '--lambda_depth_normal','0','--lambda_multi_view_ncc','.6','--lambda_multi_view_geo','.02',
            '--multi_view_ncc_views','1','--ncc_mask_thresh','.9','--quiet',*extra]
    try:
        run('warmup',training('control',common,['--warmup-stop',str(warmup)]))
        warmup_record=json.loads((common/'solver_training_integration.json').read_text())
        if not warmup_record['completed'] or warmup_record['ncc_calls']!=0:
            raise ValueError('Common warmup incomplete or already regularized.')
        arms={}
        for variant in manifest['order']:
            run(f'train_{variant}',training(variant,out/variant,
                ['--branch-sidecar',str(sidecar),'--start_checkpoint',str(checkpoint)]))
            arms[variant]=json.loads((out/variant/'solver_training_integration.json').read_text())
            if not arms[variant]['completed'] or not arms[variant]['resume_state_identical']:
                raise ValueError('Resume branch was not fully validated.')
        manifest['branch_state_verified']=all(a['initial_state_sha256']==warmup_record['initial_state_sha256'] for a in arms.values())
        if not manifest['branch_state_verified'] or len({a['first_camera'] for a in arms.values()})!=1:
            raise ValueError('Branches have mismatched initial states or first cameras.')
        manifest['initial_state_sha256']=warmup_record['initial_state_sha256']
        manifest['first_camera']=arms['control']['first_camera']
        manifest['common_checkpoint_sha256']=warmup_record['common_checkpoint_sha256']
        if not args.smoke:
            # Evaluation uses a common historical five-pass renderer in both arms.
            # Solver effects are thus isolated to training, without an extraction variant.
            for variant in manifest['order']:
                model=out/variant
                run(f'extract_{variant}',[sys.executable,'mesh_extract.py','-m',str(model),
                    '--iteration',str(iterations),'--voxel_size','.002','--trunc_voxel_multiplier','4',
                    '--depth_max','8','--num_cluster','1'])
                run(f'align_{variant}',[sys.executable,'research/align_solver_mesh.py','--model',str(model)])
                for seed in (0,1,2):
                    run(f'evaluate_{variant}_seed{seed}',[sys.executable,'research/evaluate_mesh_seeded.py',
                        '--mesh',str(model/'recon_aligned.ply'),'--scan',args.scan[4:],'--seed',str(seed),
                        '--out',str(out/f'eval_{variant}_seed{seed}')])
                run(f'images_{variant}',[sys.executable,'research/evaluate_images.py','--model',str(model),
                    '--iteration','30000','--split','train','--out',str(out/f'images_{variant}.json')])
            pairs=[]
            for seed in (0,1,2):
                values={v:json.loads((out/f'eval_{v}_seed{seed}'/'results.json').read_text()) for v in manifest['order']}
                values['seed']=seed
                values['delta_adjquad_minus_control']={k:values['adjquad4'][k]-values['control'][k]
                    for k in ('overall','mean_d2s','mean_s2d')}
                pairs.append(values)
            dump(out/'paired_mesh_metrics.json',{'rows':pairs,
                'mean_delta_adjquad_minus_control':{k:sum(r['delta_adjquad_minus_control'][k] for r in pairs)/len(pairs)
                    for k in ('overall','mean_d2s','mean_s2d')},
                'limitation':'Three evaluator seeds on one trained pair, not three training replications.'})
        manifest['status']='complete'
    except BaseException as error:
        manifest['status']='failed';manifest['error']=repr(error)
        raise
    finally:dump(out/'manifest.json',manifest)


if __name__=='__main__':
    main()
