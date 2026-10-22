"""Independent world coefficient, public depth and scaled endpoint validation."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import numpy as np
import mpmath as mp
from research.verify_csr_decode import reference
from research.benchmark_csr_decode import read,sha,save,now
from research.exact_perspective_backend import ROOT
R=ROOT/'research'

def stable_residual(nodes,source,t,scale):
    beta=mp.mpf(1);shift=mp.mpf(0);tails=[];cap=mp.mpf(float(np.float32(.99)))
    for v,row in zip(nodes,source):
        A=mp.mpf(float(row[35]));E=v['E'];alpha=v['alpha'];after=t>v['p']
        if A==0:continue
        if after:
            if A<=cap:beta*=1-A;shift-=mp.log1p(A/(1-A)*(-mp.expm1(-E/2)))
            else:beta*=1-alpha
        lg=mp.log(alpha)-((t-v['p'])*v['rho'])**2/2
        # mpmath log1p retains tiny tails; source amplitude/energy shift retains
        # effective-alpha deviations too small to subtract from nominal A.
        logtail=mp.log(-mp.log1p(-mp.exp(lg))/2)
        tails.append((logtail,after))
    constant=mp.log(mp.mpf('.5')/beta)+shift
    result=constant*mp.exp(-scale)
    for logtail,after in tails:result+=(-1 if after else 1)*mp.exp(logtail-scale)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();base=args.base.resolve();out=args.out.resolve();assert not out.exists();m=read(base/'manifest.json');s=read(base/'summary.json');assert m['status']==s['status']=='complete' and len(s['rows'])==24
    for n,h in m['source_sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in m['artifacts_sha256'].items():assert sha(base/n)==h,n
    with zipfile.ZipFile(base/'sources.zip') as z:
        for n,h in m['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    b=m['backend'];assert sha(Path(b['binary_path']))==b['binary_sha256']
    for n,h in b['generated_sha256'].items():assert sha(base/'generated'/n)==h,n
    for n in ['guarded_preprocess.cuh','shared_cache.cuh']:assert (base/'generated'/n).read_bytes()==(R/'experiments/shared_cache_v1/generated'/n).read_bytes()
    for n in ['guarded_interval.cuh','short_interval.cuh']:assert (base/'generated'/n).read_bytes()==(R/'experiments/short_certificate_v1/variant_generated'/n).read_bytes()
    for n in ['float_product.cuh','dynamic_root.cuh']:assert (base/'generated'/n).read_bytes()==(R/n).read_bytes()
    coeff=0;rays=0;certified=0;noroot=0;failures=[];rows=[];anomalies=[];selected_total=0
    for r in s['rows']:
        stem=f'scan{r["scan"]}_{r["camera"]}'
        with np.load(base/(stem+'_output.npz')) as d:pixels=d['pixels'];output=d['output']
        with np.load(R/'experiments/float_product_v1'/(stem+'_output.npz')) as d:prior=d['output']
        with np.load(R/'experiments/csr_capture_v1'/(stem+'.npz')) as d:offsets=d['offsets'];records=d['records']
        with np.load(R/'experiments/csr_decode_v1'/(stem+'_query_amp.npz')) as d:q=d['query_xy'];amp=d['amplitude']
        assert np.array_equal(pixels,np.flatnonzero(prior[:,0]==0)) and np.all(output[:,14]==np.diff(offsets)[pixels]);assert len(pixels)==r['selected_rays'];selected_total+=len(pixels)
        codes,counts=np.unique(output[:,0],return_counts=True);assert {str(int(k)):int(v) for k,v in zip(codes,counts)}==r['final_codes']
        accepted=output[:,0]==1;assert np.all(np.isfinite(output[accepted])) and np.all(output[accepted,15]==0) and np.all(output[accepted,4]<output[accepted,5]);assert np.array_equal(output[accepted,2],output[accepted,2].astype(np.float32).astype(np.float64))
        for method in [1,2,3]:
            mask=accepted&(output[:,12]==method);assert int(mask.sum())==r['certificates'][str(method)]
            if method==1:assert np.all(output[mask,6]>.5) and np.all(output[mask,9]<.5)
            else:assert np.all(output[mask,7]<0) and np.all(output[mask,8]>0)
        assert int(np.sum(np.diff(offsets)[pixels]>192))==r['counts_above192'] and int(np.diff(offsets)[pixels].sum())==r['selected_records']
        with np.load(R/'experiments/float_product_v1'/(stem+'_probes.npz')) as d:oldprobe=d['pixels'][d['output'][:,0]==0]
        chosen=set(oldprobe.tolist());chosen.update(pixels[np.linspace(0,len(pixels)-1,min(32,len(pixels)),dtype=np.int64)].tolist())
        with np.load(base/(stem+'_probes.npz')) as d:pp=d['pixels'];source=d['source'];packets=d['packets'];sections=d['sections'];po=d['output'];ri=d['record_indices'];ids=d['primitive_ids'];pq=d['query']
        assert set(pp.tolist())==chosen and len(pp)==r['probe_rays'];assert np.array_equal(po.view(np.uint64),output[np.searchsorted(pixels,pp)].view(np.uint64));assert np.array_equal(pq,q[pp]);assert np.array_equal(np.diff(sections),np.diff(offsets)[pp]);expected=np.concatenate([np.arange(offsets[p],offsets[p+1],dtype=np.int64) for p in pp]);assert np.array_equal(ri,expected) and np.array_equal(ids,records[ri,0])
        assert np.array_equal(source[:,32:34],q[np.repeat(pp,np.diff(sections))].astype(np.float64)) and np.all(source[:,34]==1) and np.array_equal(source[:,35],amp[ids].astype(np.float64));assert np.all(packets[:,0]==0) and np.isfinite(packets).all()
        cases={a['pixel']:a['case'] for a in r['original_anomalies']}
        for j,pixel in enumerate(pp):
            begin,end=int(sections[j]),int(sections[j+1]);nodes=[]
            for i in range(begin,end):
                mp.mp.dps=80;v80=reference(source[i]);mp.mp.dps=140;v=reference(source[i]);nodes.append(v)
                for key,a,b in [('p',1,2),('rho',3,4),('E',5,6),('alpha',24,25)]:
                    value=v[key];assert abs(value-v80[key])<=mp.mpf('1e-65')*max(1,abs(value));slack=mp.mpf('1e-120')*max(1,abs(value));coeff+=1
                    if not mp.mpf(float(packets[i,a]))-slack<=value<=mp.mpf(float(packets[i,b]))+slack:failures.append(dict(view=stem,pixel=int(pixel),record=i,field=key))
            o=po[j]
            if o[0]==1:
                mp.mp.dps=240;center=mp.mpf(float(o[2]))*mp.sqrt(mp.mpf(float(pq[j,0]))**2+mp.mpf(float(pq[j,1]))**2+1);tol=mp.mpf(float.fromhex('0x1.0624dd2f1a9fbp-10'));assert center-mp.mpf(float(o[4]))<=tol and mp.mpf(float(o[5]))-center<=tol
                for t,a,b,scale in [(o[4],6,7,o[10]),(o[5],8,9,o[11])]:
                    value=stable_residual(nodes,source[begin:end],mp.mpf(float(t)),mp.mpf(float(scale)))
                    if o[12]==1:value=mp.exp(-(value+mp.log(2)))
                    slack=mp.mpf('1e-110')*max(1,abs(value))
                    if not mp.mpf(float(o[a]))-slack<=value<=mp.mpf(float(o[b]))+slack:failures.append(dict(view=stem,pixel=int(pixel),field='endpoint',method=int(o[12]),endpoint=a,value=mp.nstr(value,65)))
                    if o[12]==1:assert (value>.5 if a==6 else value<.5)
                    else:assert (value<0 if a==6 else value>0)
                certified+=1
            elif o[0] in [2,3]:
                mp.mp.dps=240;beta=mp.fprod([1-v['alpha'] for v in nodes]);assert beta>=.5;noroot+=1
            if int(pixel) in cases:anomalies.append(dict(case=cases[int(pixel)],view=stem,pixel=int(pixel),code=int(o[0]),mode=int(o[1]),certificate=int(o[12]),public_depth=None if not np.isfinite(o[2]) else float(o[2]),original_public_depth=float(prior[pixel,11])))
            rays+=1
            if (j+1)%8==0:save(out.with_suffix('.progress.json'),dict(utc=now(),completed_views=len(rows),active_view=stem,active_ray=j+1,coefficients=coeff,failures=failures));print(f'{stem}: MP rays{j+1}/{len(pp)}, coefficients={coeff}',flush=True)
        rows.append(dict(view=stem,probe_rays=len(pp),probe_nodes=len(source)))
    result=dict(status='complete',checked_utc=now(),source_checks=len(m['source_sha256']),artifact_checks=len(m['artifacts_sha256']),selected_rays=selected_total,world_probe_rays=rays,world_probe_nodes=coeff//4,coefficient_checks=coeff,certified_probe_rays=certified,no_root_probe_rays=noroot,failures=failures,gate_pass=not failures,world_precisions=[80,140],scaled_endpoint_precision=240,verifier_sha256=sha(Path(__file__)),rows=rows,old_anomalies=sorted(anomalies,key=lambda r:r['case']),scope='All output/state/CSR accounting; fixed full-ray MP bounds/public-F32 end tests, not all562335 per-node MP proofs. No GT/worldVJP/training/newness claim.')
    assert selected_total==562335 and len(anomalies)==9;save(out,result);assert not failures;print(json.dumps({k:v for k,v in result.items() if k not in ['rows','old_anomalies','scope']},indent=2))
if __name__=='__main__':main()
