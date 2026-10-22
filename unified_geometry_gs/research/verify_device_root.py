"""Independent MP240 world covariance checks at new device-created brackets."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,zipfile
import numpy as np
import mpmath as mp
from research.verify_guarded_precision import world,residual

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';BASE=R/'experiments/device_root_v1';OUT=R/'results/device_root_audit_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def frame(row):
    view=np.eye(4);view[:3,:3]=row[20:29].reshape(3,3).T;view[3,:3]=row[29:32]
    return dict(mean=row[:3],scales=row[3:6],quat=row[6:10] if row[10] else None,R=row[11:20].reshape(3,3),view=view,qray=row[32:35],amplitude=row[35])

def main():
    if OUT.exists():raise FileExistsError(OUT)
    mp.mp.dps=240;audits=[]
    for folder in ('device_budget_v1','device_root_v1'):
        p=R/'experiments'/folder;m=read(p/'manifest.json');assert m['status']=='complete'
        with zipfile.ZipFile(p/'sources.zip') as z:
            for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h and hashlib.sha256(z.read(n)).hexdigest()==h
        for n,h in m['artifacts_sha256'].items():assert sha(p/n)==h
        for n,h in m['generated_sha256'].items():assert sha(p/'generated'/n)==h
        audits.append(dict(experiment=folder,sources=len(m['source_sha256']),artifacts=len(m['artifacts_sha256']),generated=len(m['generated_sha256'])))
    with np.load(BASE/'inputs.npz') as d:source=d['source'];sections=d['sections']
    with np.load(BASE/'outputs.npz') as d:packets=d['packets'];roots=d['roots']
    meta=read(BASE/'targets_posthoc_only.json');records=[];bad=[];fieldchecks=0
    tau=Fraction(1,1000);assert Fraction(float.fromhex('0x1.0624dd2f1a9fbp-10'))<tau
    for i,(s,n) in enumerate(sections):
        label=meta['sections'][i];output=roots[i];mode=int(output[1]);params=[world(frame(source[j])) for j in range(s,s+n)]
        for j,c in enumerate(params):
            p=packets[s+j,mode];assert p[0]==0
            for field,a,b in (('p',1,2),('rho',3,4),('E',5,6),('alpha',24,25)):
                value=c[field];fieldchecks+=1;allow=mp.mpf('1e-200')*max(1,abs(value))
                if not mp.mpf(float(p[a]))-allow<=value<=mp.mpf(float(p[b]))+allow:bad.append(dict(name=label['name'],layer=j,field=field))
        if output[0]==1:
            z,lo,hi=map(Fraction,output[2:5]);assert hi-lo<=2*tau and z-lo<=tau and hi-z<=tau
            L=residual(params,mp.mpf(float(output[3])));H=residual(params,mp.mpf(float(output[4])))
            if not L<0<H:bad.append(dict(name=label['name'],field='world_endpoint_sign'))
            for actual,a,b,scale in ((L,5,6,9),(H,7,8,10)):
                value=actual*mp.exp(-mp.mpf(float(output[scale])));low=mp.mpf(float(output[a]));high=mp.mpf(float(output[b]));allow=mp.mpf('1e-200')*max(abs(value),abs(low),abs(high))
                if not low-allow<=value<=high+allow:bad.append(dict(name=label['name'],field='world_scaled_enclosure'))
            target=mp.mpf(meta['targets'][label['name']]['camera_ray_depth']);assert mp.mpf(float(output[3]))<=target<=mp.mpf(float(output[4])) and abs(mp.mpf(float(output[2]))-target)<=mp.mpf('1e-3')
            record=dict(name=label['name'],mode=mode,left_H=mp.nstr(L,100),right_H=mp.nstr(H,100),width_numerator=str((hi-lo).numerator),width_denominator=str((hi-lo).denominator),classification='finite')
        else:
            exact=all(c['E']==0 for c in params);beta_exact=Fraction(1)
            if exact:
                for c in params:beta_exact*=1-c['A']
            beta=mp.fprod(1-c['alpha'] for c in params)
            classification='critical_no_finite_root' if exact and beta_exact==Fraction(1,2) else 'insufficient_mass_no_finite_root' if beta>mp.mpf('.5') else 'unresolved'
            if classification!=label['expected']:bad.append(dict(name=label['name'],field='world_mass_classification'))
            record=dict(name=label['name'],mode=mode,classification=classification,terminal=mp.nstr(beta,100))
        records.append(record)
    assert len(records)==110 and sum(r['classification']=='finite' for r in records)==106 and not bad
    result=dict(status='complete',archive_checks=audits,independent_world_dps=240,used_coefficient_checks=fieldchecks,new_device_endpoint_pairs=106,no_root_models=4,failures=bad,rows=records,
        scope='Independent covariance reference at device-created roots; no reference candidate/bracket was supplied to device. Mathematical ray models with fixed accepted prefixes, not full renderer/GT/backward/efficiency evidence.')
    OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','scope')},indent=2))

if __name__=='__main__':main()
