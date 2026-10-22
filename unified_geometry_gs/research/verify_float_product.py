"""Independent complete output accounting and predeclared full-ray MP probes."""
from pathlib import Path
from datetime import datetime,timezone
from fractions import Fraction
import argparse,hashlib,json,zipfile
import numpy as np
import mpmath as mp
from research.verify_csr_decode import reference
from research.benchmark_csr_decode import read,sha,save
from research.exact_perspective_backend import ROOT
R=ROOT/'research'

def transmittance(nodes,t):
    total=mp.mpf(1)
    for v in nodes:
        g=mp.exp(-((t-v['p'])*v['rho'])**2/2);b=mp.sqrt(1-v['alpha']*g)
        total*=b if t<=v['p'] else (1-v['alpha'])/b
    return total

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();base=args.base.resolve();out=args.out.resolve();assert not out.exists()
    m=read(base/'manifest.json');s=read(base/'summary.json');assert m['status']==s['status']=='complete' and len(s['rows'])==24
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(base/n)==h,n
    with zipfile.ZipFile(base/'sources.zip') as z:
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    b=m['backend'];assert sha(Path(b['binary_path']))==b['binary_sha256']
    for n,h in b['generated_sha256'].items():assert sha(base/'generated'/n)==h,n
    from research.csr_decode_cuda import once
    old=(R/'experiments/shared_cache_v1/generated/shared_cache.cuh').read_text(encoding='utf-8');old=old.replace('namespace gc {','namespace fc {').replace('} // namespace gc','} // namespace fc')
    a='I alpha=mul(pt(src[35]),gi::exp(mul(pt(-.5),ebox)));double cap=double(float(.99));alpha.lo=fmax(0.,fmin(cap,alpha.lo));alpha.hi=fmin(fmin(src[35],cap),alpha.hi);out[24]=alpha.lo;out[25]=alpha.hi;'
    replacement='fp::I af=fp::alpha(src[35],ebox.lo,ebox.hi);I alpha={double(af.lo),double(af.hi)};out[24]=alpha.lo;out[25]=alpha.hi;'
    assert (base/'generated/float_cache.cuh').read_text(encoding='utf-8')=='#include "float_product.cuh"\n'+once(old,a,replacement)
    mp.mp.dps=140
    with np.load(base/'scalars.npz') as d:
        for x,y in zip(d['x'],d['y']):assert mp.mpf(float(y[0]))<=mp.exp(mp.mpf(float(x)))<=mp.mpf(float(y[1]))
        scalars=len(d['x'])
    totalpixels=0;records=0;coeff=0;rays=0;certprobes=0;rows=[];failures=[]
    oldcases=read(R/'experiments/exact_outlier_capture_v3/summary.json')['rows']
    for r in s['rows']:
        name=f'scan{r["scan"]}_{r["camera"]}'
        with np.load(base/(name+'_output.npz')) as d:output=d['output']
        with np.load(R/'experiments/csr_decode_v1'/(name+'_query_amp.npz')) as d:q=d['query_xy'];amp=d['amplitude'];cs=d['counters']
        with np.load(R/'experiments/csr_capture_v1'/(name+'.npz')) as d:offsets=d['offsets'];rec=d['records']
        assert output.shape==(r['pixels'],12) and np.all(output[:,9]==np.diff(offsets)) and np.all(output[:,11]==cs[0].reshape(-1))
        codes,counts=np.unique(output[:,0],return_counts=True);assert {str(int(k)):int(v) for k,v in zip(codes,counts)}==r['codes']
        certified=output[:,0]==1;active=output[:,0]!=2;assert np.all(np.isfinite(output[active])) and np.all(output[certified,5]>.5) and np.all(output[certified,8]<.5) and np.all(output[certified,10]==0) and np.all(output[certified,1]<output[certified,2])
        selected=np.flatnonzero(cs[5].reshape(-1)==1);expected=set(selected[np.linspace(0,len(selected)-1,min(64,len(selected)),dtype=np.int64)].tolist())
        for c in oldcases:
            if c['scan']==r['scan'] and c['camera']==r['camera']:x,y=c['pixel'];expected.add(y*r['width']+x)
        with np.load(base/(name+'_probes.npz')) as d:pixels=d['pixels'];source=d['source'];packets=d['packets'];sections=d['sections'];ids=d['ids'];indices=d['records'];probeout=d['output'];pq=d['query']
        assert set(pixels.tolist())==expected and np.array_equal(pq,q[pixels]) and np.array_equal(probeout.view(np.uint64),output[pixels].view(np.uint64))
        expectedidx=np.concatenate([np.arange(offsets[p],offsets[p+1],dtype=np.int64) for p in pixels]);assert np.array_equal(indices,expectedidx) and np.array_equal(ids,rec[indices,0])
        assert np.array_equal(np.diff(sections),np.diff(offsets)[pixels]) and len(source)==len(indices)==r['probe_records']
        assert np.all(packets[:,0]==0) and np.isfinite(packets).all()
        pix_for=np.repeat(pixels,np.diff(sections));assert np.array_equal(source[:,32:34],q[pix_for].astype(np.float64)) and np.all(source[:,34]==1) and np.array_equal(source[:,35],amp[ids].astype(np.float64))
        for j,pixel in enumerate(pixels):
            begin,end=int(sections[j]),int(sections[j+1]);nodes=[]
            for i in range(begin,end):
                mp.mp.dps=80;v80=reference(source[i]);mp.mp.dps=140;v=reference(source[i]);nodes.append(v)
                for key,a,b in [('p',1,2),('rho',3,4),('E',5,6),('alpha',24,25)]:
                    value=v[key];assert abs(value-v80[key])<=mp.mpf('1e-65')*max(1,abs(value));slack=mp.mpf('1e-120')*max(1,abs(value));coeff+=1
                    if not mp.mpf(float(packets[i,a]))-slack<=value<=mp.mpf(float(packets[i,b]))+slack:failures.append(dict(view=name,pixel=int(pixel),record=i,field=key))
            o=probeout[j]
            if o[0]!=2:
                center=mp.mpf(float(o[11]))*mp.sqrt(mp.mpf(float(pq[j,0]))**2+mp.mpf(float(pq[j,1]))**2+1);assert mp.mpf(float(o[3]))<=center<=mp.mpf(float(o[4]))
                for t,a,b in [(o[1],5,6),(o[2],7,8)]:
                    value=transmittance(nodes,mp.mpf(float(t)));slack=mp.mpf('1e-120')
                    if not mp.mpf(float(o[a]))-slack<=value<=mp.mpf(float(o[b]))+slack:failures.append(dict(view=name,pixel=int(pixel),field='T',endpoint=a,value=mp.nstr(value,60)))
                if o[0]==1:
                    assert transmittance(nodes,mp.mpf(float(o[1])))>.5 and transmittance(nodes,mp.mpf(float(o[2])))<.5
                    tol=mp.mpf(float.fromhex('0x1.0624dd2f1a9fbp-10'));assert center-mp.mpf(float(o[1]))<=tol and mp.mpf(float(o[2]))-center<=tol;certprobes+=1
            rays+=1
            if (j+1)%8==0:save(out.with_suffix('.progress.json'),dict(utc=datetime.now(timezone.utc).isoformat(),completed_views=len(rows),active_view=name,active_ray=j+1,rays=rays,coefficients=coeff,failures=failures));print(f'{name} MP rays {j+1}/{len(pixels)}, coefficients={coeff}',flush=True)
        rows.append(dict(view=name,pixels=r['pixels'],probe_rays=len(pixels),probe_records=len(source)));totalpixels+=r['pixels'];records+=r['records']
    result=dict(status='complete',checked_utc=datetime.now(timezone.utc).isoformat(),source_checks=len(m['source_sha256']),artifact_checks=len(m['artifacts_sha256']),scalar_checks=scalars,full_output_pixels=totalpixels,records_accounted=records,world_probe_rays=rays,world_probe_records=coeff//4,coefficient_checks=coeff,certified_probe_rays=certprobes,independent_precisions=[80,140],failures=failures,gate_pass=not failures,rows=rows,verifier_sha256=sha(Path(__file__)),scope='All compact outputs/accounting; fixed full-ray MP probes, not all870M coefficients. Sufficient conditions only, unresolved retained; no fallback/backward/GT claim.')
    save(out,result);assert not failures,result;print(json.dumps({k:v for k,v in result.items() if k not in ['rows','scope']},indent=2))
if __name__=='__main__':main()
