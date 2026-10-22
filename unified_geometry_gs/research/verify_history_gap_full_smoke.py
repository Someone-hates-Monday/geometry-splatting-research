"""Recompute original-recipe smoke evidence from actual archived bytes/gradients."""
from pathlib import Path
import hashlib,json,sys,time,zipfile
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def local(name):return Path('C:/'+name[7:]) if sys.platform=='win32' and name.startswith('/mnt/c/') else Path(name)
def normdiff(a,b):
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    x=a.ravel();y=b.ravel();num=den=0.
    for i in range(0,x.size,524288):
        u=x[i:i+524288].astype('float64');v=y[i:i+524288].astype('float64')
        num+=np.linalg.norm(u-v)**2;den+=np.linalg.norm(v)**2
    return float(np.sqrt(num)/max(np.sqrt(den),1e-30))
def main():
    out=R/'results/history_gap_full_smoke_audit_v1.json';assert not out.exists()
    run=R/'experiments/history_gap_full_smoke_v1';m=read(run/'manifest.json');s=read(run/'summary.json')
    assert m['status']=='complete' and m['gate_pass'] and s['gate_pass']
    for name,h in m['artifacts_sha256'].items():assert sha(run/name)==h,name
    source_count=0
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h;source_count+=1
    with zipfile.ZipFile(run/'sources.zip') as z:
        for name,h in m['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    groups={};records={};archives={}
    for label in ['disabled_a','disabled_b','computed_zero','positive']:
        folder=run/label;r=records[label]=read(folder/'history_gap_training_integration.json')
        parent=read(folder/'solver_training_integration.json');hold=read(folder/'holdout_camera_integration.json')
        assert r['status']=='complete' and r['parent_guards_verified'] and not r['full_training_completed']
        assert len(r['steps'])==6 and r['intentional_diagnostic_termination']
        assert parent['error']==hold['error']=='DiagnosticComplete()' and not parent['completed']
        assert parent['resume_state_identical'] and parent['initial_state_sha256']==m['preflight']['initial_state_sha256']
        assert parent['common_checkpoint_sha256']==m['preflight']['checkpoint_sha256']
        assert parent['sidecar_sha256']==m['preflight']['sidecar_sha256']
        assert parent['ncc_calls']==hold['ncc_calls']==6 and parent['depth_render_calls']>=6
        assert hold['filter_calls']>=2 and hold['final_test_report_render_calls']==0
        assert len(hold['pool']['train_names'])==42 and len(hold['pool']['test_names'])==7
        assert not set(hold['ncc_target_names_seen'])&set(hold['pool']['test_names'])
        assert [x['iteration'] for x in r['steps']]==list(range(7000,7006))
        assert all(x['opacity_mask_unchanged_by_Adam'] and x['ncc']>0 and x['geo']>0 for x in r['steps'])
        assert r['steps'][0]['gaussians_before_backward']==676277 and r['steps'][0]['gaussians_at_optimizer']==677291
        for name,h in r['sources'].items():assert sha(ROOT/name)==h;source_count+=1
        with zipfile.ZipFile(folder/'history_gap_sources.zip') as z:
            for name,h in r['sources'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        for name,h in r['artifacts_sha256'].items():assert sha(folder/name)==h
        for b in [r['binaries']['query'],r['binaries']['csr']]:
            assert sha(local(b.get('binary_path',str(ROOT/b.get('binary_relative_path','')))))==b['binary_sha256']
        assert sha(ROOT/r['binaries']['loss']['path'])==r['binaries']['loss']['sha256']
        archives[label]=np.load(folder/'history_gap_first_gradients.npz')
    base=archives['disabled_a'];assert base.files==list(s['gradient_groups'])
    for name in base.files:
        ref=base[name];repeat=normdiff(archives['disabled_b'][name],ref)
        zero=normdiff(archives['computed_zero'][name],ref);positive=normdiff(archives['positive'][name],ref)
        stored=s['gradient_groups'][name]
        for a,b in [(repeat,stored['repeat_relative_l2']),(zero,stored['zero_relative_l2']),(positive,stored['positive_relative_l2'])]:
            assert abs(a-b)<=max(1e-15,abs(b)*1e-12),(name,a,b)
        assert zero<=max(1e-5,5*repeat)
        groups[name]=dict(repeat=repeat,zero=zero,positive=positive)
    for a in archives.values():a.close()
    for i in range(6):
        assert len({r['steps'][i]['camera'] for r in records.values()})==1
        zero=records['computed_zero']['steps'][i];positive=records['positive']['steps'][i]
        assert zero['pixels']==positive['pixels'] and len(zero['pixels'])==128 and len(set(zero['pixels']))==128
        assert zero['RNG_unchanged'] and positive['RNG_unchanged']
    for k in ['baseline_loss','rgb','ncc','geo','depth_normal']:
        assert len({r['steps'][0][k] for r in records.values()})==1
    report=dict(status='complete',audit_gate_pass=True,epoch=time.time(),source_checks=source_count,
        artifact_checks=len(m['artifacts_sha256']),run_manifest_sha256=sha(run/'manifest.json'),summary_sha256=sha(run/'summary.json'),
        gradients=groups,actual_densification=676277!=677291,updates=24,
        scope='Recomputed archived gradients/hashes, actual original NCC/geometry and camera/resume guards; no quality inference.',
        limitations=['Runtime finite checks are reported by executed wrapper; independent audit does not replay all24 updates.',
            'First gradients precede original topology change; later trajectories need not coincide.',
            'First timing interval omits forward and includes cold work; not an end-to-end speed benchmark.'])
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
