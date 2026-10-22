"""Independent artifact, source mapping and all-array statistics audit."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent;BASE=R/'experiments/history_gap_math_v3'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def relative(a,b):return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-300))
def main():
    out=R/'results/history_gap_math_audit_v3.json';assert not out.exists();m=read(BASE/'manifest.json');s=read(BASE/'real.json');synthetic=read(BASE/'synthetic.json');symbolic=read(BASE/'symbolic.json')
    assert m['status']==s['status']=='complete' and all(symbolic['checks'].values())
    for name,h in m['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in m['artifacts_sha256'].items():assert sha(BASE/name)==h,name
    with zipfile.ZipFile(BASE/'sources.zip') as z:
        for name,h in m['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
    lookup={(r['view'],r['pixel']):r for r in s['rows']};rays=nodes=rounded=anomalies=0;counts={k:0 for k in s['pass_counts']};maxima={k:0. for k in counts};maxblocks={k:{v:0. for v in ['alpha','peak','rho']} for k in counts};D=[]
    for f in sorted(BASE.glob('scan*_gradients.npz')):
        old=R/'experiments/world_root_vjp_v1'/f.name
        with np.load(old) as d:op=d['pixels'];ose=d['sections']
        with np.load(f) as d:
            assert np.array_equal(d['pixels'],op) and np.array_equal(d['sections'],ose)
            ref=d['reference'];assert ref.shape==d['coefficients'].shape==(int(ose[-1]),3)
            assert np.isfinite(ref).all() and np.isfinite(d['log_alpha']).all()
            assert np.isfinite(d['coefficients']).all() and np.all(d['coefficients'][:,0]>=0) and np.all(d['coefficients'][:,0]<1) and np.all(d['coefficients'][:,2]>0)
            stem=f.name.removesuffix('_gradients.npz')
            for j,pixel in enumerate(op):
                a,b=map(int,ose[j:j+2]);r=lookup[stem,int(pixel)];assert r['nodes']==b-a;zero=int(np.count_nonzero(d['coefficients'][a:b,0]==0));assert zero==r['rounded_alpha_zero'];rounded+=zero
                assert 0<=r['D']<np.log(2) and abs(r['loss']-r['D']**2)<2e-16
                assert abs(r['ratio'])<=1 and float(r['identity_error'])<1e-120
                for branch in counts:
                    arr=d[branch][a:b];assert np.isfinite(arr).all();v=relative(arr,ref[a:b]);blocks={key:relative(arr[:,k],ref[a:b,k]) for k,key in enumerate(['alpha','peak','rho'])};passed=v<=1e-5 and max(blocks.values())<=1e-4
                    assert v==r[branch]['relative_l2'] and blocks==r[branch]['blocks'] and passed==r[branch]['gate_pass'];counts[branch]+=passed;maxima[branch]=max(maxima[branch],v)
                    for key,value in blocks.items():maxblocks[branch][key]=max(maxblocks[branch][key],value)
                rays+=1;anomalies+=r['old_anomaly'] is not None;D.append(r['D'])
            nodes+=len(ref)
    assert rays==891 and nodes==51524 and anomalies==9 and rounded==5
    assert counts==s['pass_counts'] and maxima==s['max_relative_l2'] and counts['float64']==891
    assert len(synthetic['regular'])==16 and all(float(e)<=1e-30 for row in synthetic['regular'] for e in row['relative_fd'])
    assert len(synthetic['adverse'])==4 and all(float(e)<=1e-30 for row in synthetic['adverse'] for e in row['relative_fd'])
    result=dict(status='complete',gate_pass=True,rays=rays,nodes=nodes,source_checks=len(m['source_sha256']),artifact_checks=len(m['artifacts_sha256']),all9_old_anomalies_present=True,zero_rounded_alpha_nodes=rounded,pass_counts=counts,max_relative_l2=maxima,max_block_relative_l2=maxblocks,D_range=[min(D),max(D)],regular_max_relative_fd=max(float(e) for row in synthetic['regular'] for e in row['relative_fd']),adverse=[{k:v for k,v in row.items() if k in ['epsilon','loss','gradient_norm']} for row in synthetic['adverse']],limitations=['CPU coefficient-gradient checks with additional log-alpha input, not complete raw-chain/CUDA/training proof.','891 fixed development rays do not estimate a population failure rate.','Large-loss exponentially small-gradient counterexample remains; no quality or novelty claim.'])
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
