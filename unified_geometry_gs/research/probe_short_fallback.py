"""One fixed flat single-primitive family using frozen CUDA binaries."""
from pathlib import Path
from datetime import datetime, timezone
from fractions import Fraction
import hashlib, importlib.util, json, os, traceback, zipfile
import numpy as np
import mpmath as mp
import torch
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research';OLD=R/'experiments/short_certificate_v1';OUT=R/'experiments/short_fallback_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def finite(x):return float(x) if np.isfinite(x) else str(float(x))
def binary(folder):
    paths=list((OLD/folder).glob('*.so'));assert len(paths)==1;p=paths[0]
    spec=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m,p
def run(state,source,sections,control,variant):
    x=torch.tensor(source,dtype=torch.float64,device='cuda');sec=torch.tensor(sections,dtype=torch.int32,device='cuda')
    cp=control.preprocess(x);vp=variant.preprocess(x);cr=control.solve(cp,sec);vr=variant.solve(vp,sec);torch.cuda.synchronize()
    cp,vp,cr,vr=[a.cpu().numpy() for a in (cp,vp,cr,vr)]
    np.savez_compressed(OUT/'outputs.npz',control_packets=cp,variant_packets=vp,control_roots=cr,variant_roots=vr)
    assert np.array_equal(cp.view(np.uint64),vp.view(np.uint64))
    mp.mp.dps=240;records=[];bad=[]
    for arm,roots in [('control',cr),('variant',vr)]:
        for i,row in enumerate(roots):
            s=mp.mpf(float(source[i,3]));mode=int(row[1]);p=vp[i,mode]
            for actual,a,b in [(mp.mpf(1),1,2),(1/s,3,4),(mp.mpf(0),5,6),(mp.mpf(3)/4,24,25)]:
                if not mp.mpf(float(p[a]))<=actual<=mp.mpf(float(p[b])):bad.append(dict(arm=arm,scale=str(s),field='coefficient_enclosure'))
            okay=True
            if row[0]==1:
                z,lo,hi=map(Fraction,row[2:5]);tau=Fraction(1,1000)
                okay=bool(lo<=1<=hi and hi-lo<=2*tau and z-lo<=tau and hi-z<=tau and row[6]<0<row[7])
                for a,b,scale,endpoint,sign in [(5,6,9,3,-1),(7,8,10,4,1)]:
                    t=mp.mpf(float(row[endpoint]));g=mp.mpf(3)/4*mp.exp(-mp.mpf('.5')*((t-1)/s)**2)
                    H=-mp.log1p(-g)/2 if t<=1 else -mp.log(mp.mpf(1)/4)+mp.log1p(-g)/2
                    residual=H-mp.log(2);value=residual*mp.exp(-mp.mpf(float(row[scale])))
                    lower,upper=mp.mpf(float(row[a])),mp.mpf(float(row[b]));allow=mp.mpf('1e-200')*max(abs(value),abs(lower),abs(upper))
                    okay=okay and bool(sign*residual>0 and lower-allow<=value<=upper+allow)
            else:
                okay=bool(row[0]==0)  # Finite mathematical root cannot be mass-classified as no root.
            if not okay:bad.append(dict(arm=arm,scale=str(s),field='certificate_or_status'))
            record=dict(arm=arm,scale=float(s),output=list(map(finite,row)),returned='certified' if row[0]==1 else 'unresolved',independent_MP240_check_pass=okay)
            if arm=='variant':record.update(route='short' if row[12]==1 else 'strict' if row[12]==2 else 'unresolved',strict_attempts=int(row[13]))
            records.append(record)
    save(OUT/'rows.json',dict(rows=records))
    v=[r for r in records if r['arm']=='variant'];c=[r for r in records if r['arm']=='control']
    result=dict(status='complete',models=8,independent_precision=240,source_scale_powers=list(range(8)),failures=bad,no_false_certificate_gate_pass=not bad,
        control_certified=sum(r['returned']=='certified' for r in c),control_unresolved=sum(r['returned']=='unresolved' for r in c),
        variant_short_certified=sum(r['route']=='short' for r in v),variant_strict_certified=sum(r['route']=='strict' for r in v),variant_unresolved=sum(r['route']=='unresolved' for r in v),
        strict_success_branch_covered=any(r['route']=='strict' for r in v),scope='One fixed mathematical flat family after development, not part of original adoption/timing cohort; unresolved retained, analytic roots never fed to CUDA.')
    save(OUT/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='manifest.json'});save(OUT/'manifest.json',state)
    print(json.dumps(result,indent=2))
    if bad:raise AssertionError('Independent stress certificate check failed')
def main():
    if OUT.exists():raise FileExistsError(OUT)
    m=read(OLD/'manifest.json');assert m['status']=='complete'
    control,cb=binary('control_generated');variant,vb=binary('variant_generated')
    for p in [cb,vb]:assert sha(p)==m['generated_sha256'][p.relative_to(OLD).as_posix()]
    source=np.zeros((8,36),np.float64)
    for j in range(8):
        row=source[j];row[:3]=[0,0,1];row[3:6]=10.**j;row[6]=1;row[10]=1;row[11:20]=np.eye(3).ravel();row[20:29]=np.eye(3).ravel();row[34]=1;row[35]=.75
    sections=np.array([[j,1] for j in range(8)],np.int32)
    sources=[Path(__file__),R/'SHORT_FALLBACK_PROTOCOL.md',OLD/'manifest.json',OLD/'summary.json']+[OLD/d/n for d in ['control_generated','variant_generated'] for n in ['guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh','root.cu','binding.cpp']]+[OLD/'variant_generated/short_interval.cuh']
    OUT.mkdir();np.savez_compressed(OUT/'inputs.npz',source=source,sections=sections)
    state=dict(status='running',created_utc=now(),pid=os.getpid(),source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},binary_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [cb,vb]})
    save(OUT/'manifest.json',state)
    with zipfile.ZipFile(OUT/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT).as_posix())
    try:run(state,source,sections,control,variant)
    except BaseException as e:
        state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(OUT/'manifest.json',state);(OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
