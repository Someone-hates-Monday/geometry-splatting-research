"""Full Gram+factor backend compile and 20-update smoke, never full training."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--factor-smoke',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    base=read(args.base/'manifest.json');ref=read(args.factor_smoke/'factor_training_integration.json')
    audit=read(args.audit/'manifest.json');result=read(args.audit/'results.json')
    if base['status']!='complete' or ref['status']!='complete' or ref['steps_observed']!=20 or ref['nonfinite_gradient_groups']:raise ValueError('Complete common state and factor smoke required.')
    if ref['initial_state_sha256']!=base['initial_state_sha256'] or audit['status']!='complete' or not result['gram32_forward_gate_pass'] or result['case_count']!=164:raise ValueError('Forward admission failed.')
    if sha(args.audit/'results.json')!=audit['results_sha256'] or sha(args.audit/'sources.zip')!=audit['source_archive_sha256']:raise ValueError('Audit bytes changed.')
    with zipfile.ZipFile(args.audit/'sources.zip') as z:
        for n,h in audit['source_sha256'].items():
            if sha(ROOT/n)!=h or hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('Audited recipe changed: '+n)
    sources=base['source_sha256'].copy()
    for n in ('run_gram_smoke.py','train_gram_branch.py','gram_factor_preprocess_backend.py','gram_forward_probe.py','factor_preprocess_backend.py',
        'factor_preprocess_probe.py','stable_inverse_backend.py','precise_preprocess_backend.py','world_preprocess_probe.py','GRAM_FORWARD_SMOKE_PROTOCOL.md'):
        sources['research/'+n]=sha(ROOT/'research'/n)
    raster=ROOT/'submodules/diff-gaussian-rasterization'
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for x in raster.glob(pattern):sources[x.relative_to(ROOT).as_posix()]=sha(x)
    for x in (raster/'third_party/glm/glm').rglob('*'):
        if x.is_file():sources[x.relative_to(ROOT).as_posix()]=sha(x)
    for n,h in sources.items():
        if sha(ROOT/n)!=h:raise ValueError('Base source changed: '+n)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    from research.gram_factor_preprocess_backend import load_backend,kernel_identity,payloads
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n in sources:z.write(ROOT/n,n)
        for n,v in payloads().items():z.writestr('generated/gram_factor64/'+n,v)
    state={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'building','smoke_only':True,'requested_updates':20,
        'source_sha256':sources,'source_archive_sha256':sha(out/'sources.zip'),'audit_manifest_sha256':sha(args.audit/'manifest.json'),
        'reference_integration_sha256':sha(args.factor_smoke/'factor_training_integration.json'),
        'first_loss_scaled_gate':1e-4,'first_loss_absolute_floor':1e-5,'full_quality_training_started':False,
        'limits':['Forward arithmetic change plus already-audited factor64 backward.','20 updates do not establish quality or speed.',
            'Tiny smoke runs concurrently with the separate factor64 quality pilot; no timing comparison.']}
    def write():(out/'manifest.json').write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    def guard():
        for n,h in sources.items():
            if sha(ROOT/n)!=h:raise ValueError('Queued smoke source changed: '+n)
    write()
    try:
        guard();mod=load_backend();state['backend_sha256']=sha(Path(mod.__file__));state['kernel_identity']=kernel_identity(mod);state['status']='running';write()
        original=next(x['argv'] for x in base['steps'] if x['stage']=='train_control');train_args=original[2:].copy()
        if train_args[:2]!=['--solver','control']:raise ValueError('Base argv changed.')
        train_args=train_args[2:];model=out/'gram_factor64';train_args[train_args.index('-m')+1]=str(model)
        argv=[sys.executable,'research/train_gram_branch.py','--world-variant','gram_factor64','--diagnostic-steps','20',*train_args]
        state['argv']=argv;write();guard()
        with (out/'train_gram_factor64.log').open('w') as log:rc=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        state['returncode']=rc;write()
        if rc:raise RuntimeError('Full backend smoke failed; keep all outputs.')
        rec=read(model/'gram_training_integration.json')
        if rec['status']!='complete' or rec['steps_observed']!=20 or rec['nonfinite_gradient_groups'] or rec['backend_sha256']!=state['backend_sha256']:raise ValueError('Smoke completion/binary mismatch.')
        for k in ('initial_state_sha256','common_checkpoint_sha256','sidecar_sha256','first_camera','initial_gaussians'):
            if rec[k]!=ref[k]:raise ValueError('Common state differs: '+k)
        if set(rec['first_loss_scalars'])!=set(ref['first_loss_scalars']):raise ValueError('Loss schema differs.')
        errors={k:abs(v-ref['first_loss_scalars'][k])/max(abs(ref['first_loss_scalars'][k]),1e-5) for k,v in rec['first_loss_scalars'].items()}
        state['first_loss_scaled_errors']=errors;state['first_loss_gate_pass']=max(errors.values())<=1e-4
        state['integration_sha256']=sha(model/'gram_training_integration.json')
        if not state['first_loss_gate_pass']:raise ValueError('First-loss engineering gate failed; no full training.')
        state['status']='complete';print(json.dumps({'status':'complete','first_loss_scaled_errors':errors,'full_quality_training_started':False},indent=2),flush=True)
    except BaseException as e:state['status']='failed';state['error']=repr(e);raise
    finally:state['updated_utc']=datetime.now(timezone.utc).isoformat();write()
if __name__=='__main__':main()
