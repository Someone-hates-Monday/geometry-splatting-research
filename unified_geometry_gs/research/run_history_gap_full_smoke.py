"""Four independent original-recipe runs, then CPU comparison of first gradients."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time,traceback,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def relative(a,b):
    assert a.shape==b.shape
    a=a.reshape(-1);b=b.reshape(-1);num=0.;den=0.
    for start in range(0,len(a),1048576):
        aa=a[start:start+1048576].astype(np.float64);bb=b[start:start+1048576].astype(np.float64)
        num+=float(np.square(aa-bb).sum());den+=float(np.square(bb).sum())
    return float(np.sqrt(num)/max(np.sqrt(den),1e-30))
def recipe(model):
    common=R/'experiments/factor_holdout55_v2/common'
    return ['-s',str(ROOT.parent/'dtu/DTU/scan55'),'-m',str(model),'-r','2','--dataset','dtu','--eval',
        '--use_decoupled_appearance','3','--iterations','30000','--regularization_from_iter','7000',
        '--save_iterations','30000','--test_iterations','30000','--checkpoint_iterations','30000',
        '--lambda_depth_normal','0','--lambda_multi_view_ncc','.6','--lambda_multi_view_geo','.02',
        '--multi_view_ncc_views','1','--ncc_mask_thresh','.9','--quiet',
        '--branch-sidecar',str(common/'chkpnt6999.runtime.pth'),'--start_checkpoint',str(common/'chkpnt6999.pth')]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True)
    preflight=read(R/'results/history_gap_full_train_preflight_v1.json');assert preflight['status']=='complete'
    sources=[Path(__file__),R/'train_history_gap_branch.py',R/'HISTORY_GAP_FULL_TRAIN_PROTOCOL.md']
    m=dict(status='running',pid=os.getpid(),epoch=time.time(),jobs=[],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},preflight=preflight)
    save(out/'manifest.json',m)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in sources:z.write(path,path.relative_to(ROOT).as_posix())
    try:
        for label,weight in [('disabled_a','disabled'),('disabled_b','disabled'),('computed_zero','0'),('positive','.01')]:
            model=out/label
            argv=[sys.executable,'-u','-m','research.train_history_gap_branch','--hg-label',label,
                '--hg-weight',weight,'--hg-updates','6',*recipe(model)]
            job=dict(label=label,status='running',argv=argv,started_epoch=time.time());m['jobs'].append(job)
            with (out/(label+'.log')).open('w') as log:
                child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                job['pid']=child.pid;save(out/'manifest.json',m)
                print(f'{label} running pid={child.pid}',flush=True)
                code=child.wait()
            job.update(status='complete' if code==0 else 'failed',returncode=code,wall_seconds=time.time()-job['started_epoch'])
            save(out/'manifest.json',m)
            print(f'{label} exit={code} wall={job["wall_seconds"]:.2f}s',flush=True)
            assert code==0,(label,(out/(label+'.log')).read_text()[-3000:])
        records={x:read(out/x/'history_gap_training_integration.json') for x in ['disabled_a','disabled_b','computed_zero','positive']}
        for rec in records.values():
            assert rec['status']=='complete' and rec['parent_guards_verified'] and len(rec['steps'])==6
            assert rec['initial_state_sha256']==preflight['initial_state_sha256']
        camera_orders={k:[r['camera'] for r in v['steps']] for k,v in records.items()}
        assert all(x==camera_orders['disabled_a'] for x in camera_orders.values())
        assert all(records['computed_zero']['steps'][i]['pixels']==records['positive']['steps'][i]['pixels'] for i in range(6))
        gradients={k:np.load(out/k/'history_gap_first_gradients.npz') for k in records}
        assert all(set(d.files)==set(gradients['disabled_a'].files) for d in gradients.values())
        result={}
        for name in gradients['disabled_a'].files:
            ref=gradients['disabled_a'][name]
            repeat=relative(gradients['disabled_b'][name],ref);zero=relative(gradients['computed_zero'][name],ref)
            effect=relative(gradients['positive'][name],ref)
            result[name]=dict(repeat_relative_l2=repeat,zero_relative_l2=zero,positive_relative_l2=effect,
                threshold=max(1e-5,5*repeat),gate_pass=zero<=max(1e-5,5*repeat))
        none=[r['steps'][0]['first_gradients_absent'] for r in records.values()]
        assert all(set(x)==set(none[0]) for x in none)
        gate=all(x['gate_pass'] for x in result.values())
        summary=dict(status='complete',gate_pass=gate,camera_orders=camera_orders,gradient_groups=result,
            first_gradients_absent=none[0],initial_state_sha256=preflight['initial_state_sha256'],
            final_gaussians={k:r['steps'][-1]['gaussians_at_optimizer'] for k,r in records.items()},
            steps={k:r['steps'] for k,r in records.items()},
            scope='Six original full-recipe updates per independent process; first gradient gate before topology changes. No heldout or GT inference.')
        save(out/'summary.json',summary)
        for d in gradients.values():d.close()
        m.update(status='complete',gate_pass=gate,artifacts_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['manifest.json','events.out.tfevents'] and not p.name.startswith('events.out.tfevents')})
        save(out/'manifest.json',m);print(json.dumps(dict(status=m['status'],gate_pass=gate,groups=result,final_gaussians=summary['final_gaussians'])),flush=True)
        assert gate,'Computed-zero first-gradient engineering gate failed'
    except BaseException as exc:
        m.update(status='failed',failure=repr(exc));save(out/'manifest.json',m)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
