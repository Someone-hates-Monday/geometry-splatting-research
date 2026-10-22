"""Preregistered smoke or matched-precision reconstruction pilot for local VJP fixes."""
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
sys.path.insert(0,str(ROOT))


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def verify_audit(path):
    state=json.loads((path/'manifest.json').read_text());result=json.loads((path/'results.json').read_text())
    if state['status']!='complete' or sha(path/'results.json')!=state['results_sha256'] or sha(path/'sources.zip')!=state['source_archive_sha256']:
        raise ValueError('Audit incomplete or bytes changed.')
    with zipfile.ZipFile(path/'sources.zip') as z:
        for name,digest in state['source_sha256'].items():
            if sha(ROOT/name)!=digest or hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Audit source changed: '+name)
    if any(r['failed_groups'] for r in result['summary']['all64'].values()):raise ValueError('Independent all64 derivative gate failed.')
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-result',type=Path);args=p.parse_args()
    base=json.loads((args.base/'manifest.json').read_text())
    if base['status']!='complete' or not base['branch_state_verified']:raise ValueError('Verified common state required.')
    audits=[ROOT/'research/experiments'/name for name in ('world_preprocess_synthetic_v2','world_preprocess_real_v1')]
    for audit in audits:verify_audit(audit)
    if not args.smoke:
        if args.smoke_result is None:raise ValueError('Completed smoke result required.')
        smoke=json.loads((args.smoke_result/'manifest.json').read_text())
        if smoke['status']!='complete' or not smoke['smoke_only'] or not smoke['first_forward_losses_identical'] or not smoke['branch_state_verified']:
            raise ValueError('Smoke prerequisite failed.')
        for n,s in smoke['source_sha256'].items():
            if sha(ROOT/n)!=s:raise ValueError('Smoke source changed: '+n)
    sources=base['source_sha256'].copy()
    for n,s in sources.items():
        if sha(ROOT/n)!=s:raise ValueError('Original source changed: '+n)
    for name in ('run_corrected_quality_pair.py','train_corrected_branch.py','corrected_preprocess_backend.py',
        'world_preprocess_probe.py','precise_preprocess_backend.py','audit_world_preprocess.py','audit_world_preprocess_real.py',
        'verify_preprocess_chain_rules.py','CORRECTED_PREPROCESS_PROTOCOL.md','evaluate_images.py',
        'align_solver_mesh.py','evaluate_mesh_seeded.py'):
        path=ROOT/'research'/name;sources[path.relative_to(ROOT).as_posix()]=sha(path)
    raster=ROOT/'submodules/diff-gaussian-rasterization'
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for x in raster.glob(pattern):sources[x.relative_to(ROOT).as_posix()]=sha(x)
    for x in (raster/'third_party/glm/glm').rglob('*'):
        if x.is_file():sources[x.relative_to(ROOT).as_posix()]=sha(x)
    for name in ('mesh_extract.py','dtu_eval/eval.py','gaussian_renderer/__init__.py','utils/loss_utils.py','scene/gaussian_model.py'):
        sources[name]=sha(ROOT/name)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    from research.corrected_preprocess_backend import payloads
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in sources:z.write(ROOT/n,n)
        for n,v in payloads('corrected64').items():z.writestr('generated/corrected64/'+n,v)
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'running','steps':[],
        'smoke_only':args.smoke,'scan':base['scan'],'common_initial_state_sha256':base['initial_state_sha256'],
        'source_sha256':sources,'source_archive_sha256':sha(out/'sources.zip'),
        'audit_manifests_sha256':{str(x):sha(x/'manifest.json') for x in audits},
        'order':['legacy64','corrected64'],'target_iteration':30000,'smoke_optimizer_updates':20,
        'pilot_gate':{'mean_overall_candidate_control_ratio_le':.99,'mean_s2d_delta_le':0.,'train_psnr_delta_ge':-.2},
        'limits':['One shared-state scan55 pair, not independent training replications or a significance test.',
                  'Training views only for image metrics; no held-out image claim.',
                  'Sequential wall clock is resource accounting, not a speed comparison.',
                  'Local derivative oracle does not certify full renderer or hard masks.']}
    if args.smoke_result:state['smoke_manifest_sha256']=sha(args.smoke_result/'manifest.json')
    def write():dump(out/'manifest.json',state)
    write()
    def guard():
        for n,s in sources.items():
            if sha(ROOT/n)!=s:raise ValueError('Queued source changed: '+n)
    def run(stage,cmd):
        guard();entry={'stage':stage,'status':'running','argv':cmd};state['steps'].append(entry);write()
        print(f'[{stage}] starting',flush=True);start=time.perf_counter()
        with (out/f'{stage}.log').open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        entry.update({'status':'complete' if rc==0 else 'failed','returncode':rc,'wall_seconds':time.perf_counter()-start});write()
        if rc:raise RuntimeError(stage+' failed; logs and sources retained.')
    try:
        from research.corrected_preprocess_backend import load_backend,kernel_identity
        state['backend_preflight']={}
        for v in state['order']:
            guard();module=load_backend(v)
            state['backend_preflight'][v]={'binary_sha256':sha(Path(module.__file__)),'kernel_identity':kernel_identity(module,v)};write()
        original=next(x['argv'] for x in base['steps'] if x['stage']=='train_control')
        train_args=original[2:].copy()
        if train_args[:2]!=['--solver','control']:raise ValueError('Original branch argv changed.')
        train_args=train_args[2:];integrations={}
        for v in state['order']:
            opts=['--diagnostic-steps','20'] if args.smoke else []
            argv=[sys.executable,'research/train_corrected_branch.py','--world-variant',v,*opts,*train_args]
            argv[argv.index('-m')+1]=str(out/v);run('train_'+v,argv)
            integration=json.loads((out/v/'corrected_training_integration.json').read_text());integrations[v]=integration
            if integration['status']!='complete' or integration['initial_state_sha256']!=base['initial_state_sha256'] or integration['first_camera']!=base['first_camera']:
                raise ValueError('Complete-state branch identity failed.')
            if integration['backend_sha256']!=state['backend_preflight'][v]['binary_sha256']:raise ValueError('Loaded backend differs from preflight.')
            if args.smoke and integration['nonfinite_gradient_groups']:raise ValueError('Nonfinite smoke gradient.')
        a,b=[integrations[v] for v in state['order']]
        for key in ('common_checkpoint_sha256','sidecar_sha256','initial_state_sha256','first_camera','initial_gaussians'):
            if a[key]!=b[key]:raise ValueError('Paired initial state differs: '+key)
        state['branch_state_verified']=True
        state['first_forward_losses_identical']=a['first_loss_scalars']==b['first_loss_scalars']
        if not state['first_forward_losses_identical']:raise ValueError('First forward loss components differ.')
        write()
        if not args.smoke:
            metrics={};images={}
            for v in state['order']:
                model=out/v
                run('extract_'+v,[sys.executable,'mesh_extract.py','-m',str(model),'--iteration','30000',
                    '--voxel_size','.002','--trunc_voxel_multiplier','4','--depth_max','8','--num_cluster','1'])
                run('align_'+v,[sys.executable,'research/align_solver_mesh.py','--model',str(model)])
                metrics[v]=[]
                for seed in (0,1,2):
                    run(f'eval_{v}_seed{seed}',[sys.executable,'research/evaluate_mesh_seeded.py',
                        '--mesh',str(model/'recon_aligned.ply'),'--scan',base['scan'][4:],'--seed',str(seed),
                        '--out',str(out/f'eval_{v}_seed{seed}')])
                    metrics[v].append(json.loads((out/f'eval_{v}_seed{seed}'/'results.json').read_text()))
                run('images_'+v,[sys.executable,'research/evaluate_images.py','--model',str(model),'--iteration','30000',
                    '--split','train','--out',str(out/f'images_{v}.json')])
                images[v]=json.loads((out/f'images_{v}.json').read_text())
            delta={k:sum(b[k]-a[k] for a,b in zip(metrics['legacy64'],metrics['corrected64']))/3 for k in ('overall','mean_d2s','mean_s2d')}
            ratio=sum(x['overall'] for x in metrics['corrected64'])/sum(x['overall'] for x in metrics['legacy64'])
            # Image schema is inspected explicitly; fail instead of guessing a
            # metric key when a future evaluator changes its output contract.
            psnr=images['corrected64']['mean']['psnr_global_rgb_db']-images['legacy64']['mean']['psnr_global_rgb_db']
            result={'mesh_metrics':metrics,'mean_delta_corrected_minus_legacy':delta,'mean_overall_ratio':ratio,
                'train_psnr_delta':psnr,'pilot_gate_pass':ratio<=.99 and delta['mean_s2d']<=0 and psnr>=-.2,
                'limits':state['limits']}
            dump(out/'paired_quality.json',result);state['quality_results_sha256']=sha(out/'paired_quality.json');state['pilot_gate_pass']=result['pilot_gate_pass']
        state['status']='complete'
    except BaseException as e:state['status']='failed';state['error']=repr(e);raise
    finally:write()


if __name__=='__main__':main()
