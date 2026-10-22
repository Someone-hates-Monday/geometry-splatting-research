"""Recompute every post-hoc metric from all84 immutable per-camera arrays."""
from pathlib import Path
import hashlib,json,math,time,zipfile
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent;ROOT=R.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    out=R/'results/history_gap_posthoc_audit_v1.json';assert not out.exists()
    run=R/'experiments/history_gap_posthoc_v1';m=read(run/'manifest.json');s=read(run/'summary.json')
    assert m['status']==s['status']=='complete'
    for name,h in m['sources'].items():assert sha(ROOT/name)==h
    for name,h in m['artifacts_sha256'].items():assert sha(run/name)==h
    with zipfile.ZipFile(run/'sources.zip') as z:
        for name,h in m['sources'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    pilot=R/'experiments/history_gap_pilot_v1';pm=read(pilot/'manifest.json');assert sha(pilot/'manifest.json')==m['pilot_manifest_sha256']
    combined={};arrays={}
    for label in ['control','history_gap']:
        values=[];active=[];infos=[]
        assert sha(pilot/label/'point_cloud/iteration_30000/point_cloud.ply')==m['model_inputs'][label]['ply_sha256']
        for cam in pm['train_names']:
            with np.load(run/f'{label}_{cam}.npz') as d:
                x=d['values'];a=d['active'];info=d['info'];pixels=d['pixels'];counts=d['counts']
                assert len(x)==len(a)==len(info)==len(pixels)==len(counts)==512
                assert np.array_equal(a,counts>0) and np.isfinite(x).all() and np.isfinite(info[a]).all()
                assert np.isnan(info[~a]).all() and np.array_equal(x[~a],np.full((~a).sum(),math.log(4)**2))
                # Actual DTU inputs here are cropped, not a nominal1600x1200.
                with Image.open(ROOT.parent/'dtu/DTU/scan55/images'/(cam+'.png')) as image:
                    width,height=image.size
                expected=np.random.default_rng(np.random.SeedSequence([13024,int(cam)])).choice(round(width/2)*round(height/2),512,replace=False)
                assert np.array_equal(pixels,expected)
                assert np.array_equal(x[a],np.square(info[a,2]))
                values.extend(x);active.extend(a);infos.extend(info)
        x=np.asarray(values);a=np.asarray(active);info=np.asarray(infos);arrays[label]=(a,x,info)
        raw=s['combined'][label]
        stats=dict(samples=len(x),empty_fraction=float((~a).mean()),full_mean=float(np.mean(x)),
            empty_constant_contribution=float((~a).mean()*math.log(4)**2),
            nonempty_conditional_mean=float(x[a].mean()),nonempty_full_denominator_contribution=float(x[a].sum()/len(x)))
        for k,v in stats.items():assert abs(v-raw[k])<1e-12
        combined[label]=stats
    a,x,_=arrays['control'];b,y,_=arrays['history_gap'];common=a&b
    paired=dict(common_nonempty=int(common.sum()),control_mean=float(x[common].mean()),history_gap_mean=float(y[common].mean()),mean_delta=float((y[common]-x[common]).mean()),full_mean_delta=float((y-x).mean()),
        empty_to_nonempty=int((~a&b).sum()),nonempty_to_empty=int((a&~b).sum()),both_empty=int((~a&~b).sum()))
    for k,v in paired.items():assert abs(v-s['paired'][k])<1e-12
    result=dict(status='complete',audit_gate_pass=True,epoch=time.time(),arrays_checked=84,samples_per_model=len(x),combined=combined,paired=paired,
        common_nonempty_relative_loss_drop=1-paired['history_gap_mean']/paired['control_mean'],
        manifest_sha256=sha(run/'manifest.json'),summary_sha256=sha(run/'summary.json'),
        scope='Recomputed sampled-pixel identities, counts, empty constants, info-D squared losses and all means. Not independent native recapture, root certification or causal inference.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
