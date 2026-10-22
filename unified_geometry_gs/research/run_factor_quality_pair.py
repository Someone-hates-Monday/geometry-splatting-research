"""Guarded direct-factor pilot with the preselected completed FP64 control."""
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
def dump(p,x):
    temporary=p.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
    temporary.replace(p)
def read(p):return json.loads(p.read_text())
def archive_verified(path):
    state=read(path/'manifest.json')
    if sha(path/'sources.zip')!=state['source_archive_sha256']:raise ValueError('Source archive changed: '+str(path))
    with zipfile.ZipFile(path/'sources.zip') as z:
        for n,h in state['source_sha256'].items():
            if hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Archived source byte mismatch: '+n)
    return state
def verify_audit(path):
    state=archive_verified(path);result=read(path/'results.json')
    if state['status']!='complete' or sha(path/'results.json')!=state['results_sha256'] or not result['factor64_primary_gate_pass']:
        raise ValueError('Factor derivative gate failed.')
    recipe=('factor_preprocess_probe.py','stable_inverse_backend.py','world_preprocess_probe.py','precise_preprocess_backend.py','audit_world_preprocess.py')
    for n in recipe:
        key='research/'+n
        if sha(ROOT/key)!=state['source_sha256'][key]:raise ValueError('Audited recipe changed: '+key)
    return state
def verify_integration(candidate,control,diagnostic,expected_binary):
    if candidate['status']!='complete' or candidate['backend_sha256']!=expected_binary:raise ValueError('Candidate backend/completion mismatch.')
    for key in ('common_checkpoint_sha256','sidecar_sha256','initial_state_sha256','first_camera','initial_gaussians','first_loss_scalars'):
        if candidate[key]!=control[key]:raise ValueError('Matched branch state differs: '+key)
    if diagnostic and (candidate['steps_observed']!=20 or candidate['nonfinite_gradient_groups']):raise ValueError('Smoke failed.')
    if not diagnostic and candidate['steps_observed']!=23000:raise ValueError('Full update count differs.')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--control-pair',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    base=read(args.base/'manifest.json');control_pair=args.control_pair.resolve();control=control_pair/'legacy64'
    old=archive_verified(control_pair);deviation=read(control_pair/'protocol_deviation_factor_v1.json')
    integration=read(control/'corrected_training_integration.json');ply=control/'point_cloud/iteration_30000/point_cloud.ply'
    if base['status']!='complete' or not base['branch_state_verified'] or old['status']!='superseded_before_candidate':raise ValueError('Base/queue prerequisite failed.')
    if integration['status']!='complete' or integration['steps_observed']!=23000 or deviation['candidate_started'] or (control_pair/'corrected64').exists():raise ValueError('Control/candidate history mismatch.')
    if sha(ply)!=deviation['control_model_sha256'] or sha(control/'corrected_training_integration.json')!=deviation['control_integration_sha256']:raise ValueError('Retained control bytes differ.')
    if integration['initial_state_sha256']!=base['initial_state_sha256']:raise ValueError('Common initial state differs.')
    audits=[ROOT/'research/experiments'/n for n in ('factor_preprocess_stress_v1','factor_preprocess_regular_v1')]
    for audit in audits:verify_audit(audit)
    sources=base['source_sha256'].copy()
    for n,h in sources.items():
        if sha(ROOT/n)!=h:raise ValueError('Original source changed: '+n)
    names=('run_factor_quality_pair.py','train_factor_branch.py','factor_preprocess_backend.py','factor_preprocess_probe.py',
        'stable_inverse_backend.py','world_preprocess_probe.py','precise_preprocess_backend.py','corrected_preprocess_backend.py',
        'train_corrected_branch.py','audit_world_preprocess.py','FACTOR_PREPROCESS_PROTOCOL.md','align_solver_mesh.py','evaluate_mesh_seeded.py','evaluate_images.py')
    for n in names:sources['research/'+n]=sha(ROOT/'research'/n)
    raster=ROOT/'submodules/diff-gaussian-rasterization'
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for x in raster.glob(pattern):sources[x.relative_to(ROOT).as_posix()]=sha(x)
    for x in (raster/'third_party/glm/glm').rglob('*'):
        if x.is_file():sources[x.relative_to(ROOT).as_posix()]=sha(x)
    for n in ('mesh_extract.py','dtu_eval/eval.py','gaussian_renderer/__init__.py','utils/loss_utils.py','scene/gaussian_model.py'):sources[n]=sha(ROOT/n)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    from research.factor_preprocess_backend import payloads,load_backend,kernel_identity
    generated=payloads()
    # Generated full-backend forward/config bytes must match original files.
    for n,v in generated.items():
        if n!='cuda_rasterizer/render_backward.cu' and v!=(raster/n).read_text():raise ValueError('Unexpected non-backward change: '+n)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in sources:z.write(ROOT/n,n)
        for n,v in generated.items():z.writestr('generated/factor64/'+n,v)
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'preflight','steps':[],
        'scan':base['scan'],'common_initial_state_sha256':base['initial_state_sha256'],
        'source_sha256':sources,'source_archive_sha256':sha(out/'sources.zip'),
        'control_reference':str(control),'control_integration_sha256':sha(control/'corrected_training_integration.json'),
        'control_model_sha256_before_candidate':sha(ply),'protocol_deviation_sha256':sha(control_pair/'protocol_deviation_factor_v1.json'),
        'audit_manifest_sha256':{str(x):sha(x/'manifest.json') for x in audits},
        'target_iteration':30000,'pilot_gate':{'mean_overall_candidate_control_ratio_le':.99,'mean_s2d_delta_le':0.,'train_psnr_delta_ge':-.2},
        'limits':['Combined intervention, not single-change ablation.','One scan55 pair; evaluator seeds are not training replicates.',
                  'Images are 49 training views, not held-out views.','Sequential wall clock is resource accounting, not a speed test.',
                  'Local VJP gates do not certify full-renderer derivatives or all forward values.']}
    def write():dump(out/'manifest.json',state)
    def guard():
        for n,h in sources.items():
            if sha(ROOT/n)!=h:raise ValueError('Queued source changed: '+n)
        if sha(control/'corrected_training_integration.json')!=state['control_integration_sha256']:raise ValueError('Control integration changed.')
        for x,h in state['audit_manifest_sha256'].items():
            if sha(Path(x)/'manifest.json')!=h:raise ValueError('Audit changed.')
    def run(stage,cmd):
        guard();entry={'stage':stage,'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'argv':cmd};state['steps'].append(entry)
        state['current_stage']=stage;write();print('['+stage+'] starting',flush=True);start=time.perf_counter()
        with (out/(stage+'.log')).open('w') as log:rc=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        entry.update(status='complete' if rc==0 else 'failed',returncode=rc,wall_seconds=time.perf_counter()-start,finished_utc=datetime.now(timezone.utc).isoformat())
        write();print('['+stage+'] exit '+str(rc),flush=True)
        if rc:raise RuntimeError(stage+' failed; retained all outputs.')
    write()
    try:
        guard();module=load_backend();state['factor_backend']={'binary_sha256':sha(Path(module.__file__)),'kernel_identity':kernel_identity(module)};write()
        from research.corrected_preprocess_backend import load_backend as oldload
        legacy=oldload('legacy64')
        if sha(Path(legacy.__file__))!=integration['backend_sha256']:raise ValueError('Control preflight binary differs.')
        state['control_backend_reverified']=True;state['status']='running';write()
        original=next(x['argv'] for x in base['steps'] if x['stage']=='train_control')
        train_args=original[2:].copy()
        if train_args[:2]!=['--solver','control']:raise ValueError('Training argv changed.')
        train_args=train_args[2:]
        for label,steps in (('factor64_smoke',20),('factor64',0)):
            argv=[sys.executable,'research/train_factor_branch.py','--world-variant','factor64',*(['--diagnostic-steps','20'] if steps else []),*train_args]
            argv[argv.index('-m')+1]=str(out/label);run('train_'+label,argv)
            rec=read(out/label/'factor_training_integration.json');verify_integration(rec,integration,bool(steps),state['factor_backend']['binary_sha256'])
            state[label+'_integration_sha256']=sha(out/label/'factor_training_integration.json');write()
        state['branch_state_verified']=True;state['first_forward_losses_identical']=True;write()
        metrics={};images={};artifacts={}
        for label,model in (('legacy64',control),('factor64',out/'factor64')):
            run('extract_'+label,[sys.executable,'mesh_extract.py','-m',str(model),'--iteration','30000','--voxel_size','.002',
                '--trunc_voxel_multiplier','4','--depth_max','8','--num_cluster','1'])
            run('align_'+label,[sys.executable,'research/align_solver_mesh.py','--model',str(model)])
            metrics[label]=[]
            for seed in (0,1,2):
                dest=out/f'eval_{label}_seed{seed}'
                run(f'eval_{label}_seed{seed}',[sys.executable,'research/evaluate_mesh_seeded.py','--mesh',str(model/'recon_aligned.ply'),
                    '--scan',base['scan'][4:],'--seed',str(seed),'--out',str(dest)])
                metrics[label].append(read(dest/'results.json'));artifacts[str(dest/'results.json')]=sha(dest/'results.json')
            image=out/f'images_{label}.json'
            run('images_'+label,[sys.executable,'research/evaluate_images.py','--model',str(model),'--iteration','30000','--split','train','--out',str(image)])
            images[label]=read(image);artifacts[str(image)]=sha(image)
            artifacts[str(model/'recon_aligned.ply')]=sha(model/'recon_aligned.ply')
            artifacts[str(model/'point_cloud/iteration_30000/point_cloud.ply')]=sha(model/'point_cloud/iteration_30000/point_cloud.ply')
        delta={k:sum(b[k]-a[k] for a,b in zip(metrics['legacy64'],metrics['factor64']))/3 for k in ('overall','mean_d2s','mean_s2d')}
        ratio=sum(x['overall'] for x in metrics['factor64'])/sum(x['overall'] for x in metrics['legacy64'])
        psnr=images['factor64']['mean']['psnr_global_rgb_db']-images['legacy64']['mean']['psnr_global_rgb_db']
        result={'mesh_metrics':metrics,'mean_delta_factor_minus_legacy':delta,'mean_overall_ratio':ratio,'train_psnr_delta':psnr,
            'pilot_gate_pass':ratio<=.99 and delta['mean_s2d']<=0 and psnr>=-.2,'artifact_sha256':artifacts,'limits':state['limits']}
        dump(out/'paired_quality.json',result);state['quality_results_sha256']=sha(out/'paired_quality.json');state['pilot_gate_pass']=result['pilot_gate_pass']
        state['status']='complete';state['current_stage']=None
    except BaseException as e:state['status']='failed';state['error']=repr(e);raise
    finally:state['updated_utc']=datetime.now(timezone.utc).isoformat();write()
if __name__=='__main__':main()
