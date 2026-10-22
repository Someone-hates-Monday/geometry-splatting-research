"""Exact rational audit independent of CUDA arithmetic; mpmath power is numerical only."""
import hashlib,json,struct,traceback
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import sympy as sp
import mpmath as mp
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compensated_edge_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(x):return F.from_float(float(x))
def rn(x):
    f=np.float32(float(x));c=[np.nextafter(f,np.float32(-np.inf)),f,np.nextafter(f,np.float32(np.inf))]
    return min(c,key=lambda a:(abs(q(a)-x),int(a.view(np.uint32))&1))
def gate(c,a,xy):
    values=np.r_[c,a]
    return bool(np.isfinite(values).all() and np.isfinite(xy).all() and all(v==0 or 2.**-60<=abs(v)<=2.**20 for v in values) and all(v==int(v) and abs(v)<=8192 for v in xy))
def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    plan=json.loads((OUT/'input_plan.json').read_text());native=json.loads((OUT/'native_results.json').read_text());assert native['execution_plan_sha256']==sha(OUT/'execution_plan.json')
    assert sha(ROOT/'TGS_COMPENSATED_EDGE_PROTOCOL_V6.md')==plan['protocol_sha256']
    assert sha(ROOT/'tgs_compensated_edge_probe.cu')==plan['source_sha256']
    for p,h in plan['files'].items():assert sha(OUT/p)==h
    for p,h in plan['used_sources'].items():assert sha(ROOT/p)==h
    with np.load(OUT/'inputs.npz') as z:inp={k:z[k] for k in z.files}
    labels=json.loads((OUT/'labels.json').read_text());n=plan['rows'];m=plan['power_rows'];mp.mp.dps=100
    c,a,xy,cache=[inp[k] for k in ['coeff','split','xy','cache']]
    # Exact cache construction, domain and binary serialization, not trusted from prepare script.
    with (OUT/'input.bin').open('rb') as f:
        assert struct.unpack('<qq',f.read(16))==(n,m)
        for arr in [c,a,xy,cache,inp['power']]:assert f.read(arr.nbytes)==arr.tobytes()
        assert not f.read()
    domain=np.array([gate(*v) for v in zip(c,a,xy)]);assert np.array_equal(domain,inp['gate'])
    B=[F(8192),F(8192),F(1)];records=[];cq=[];aq=[];exact=[];splitdot=[]
    for i in range(n):
        C=list(map(q,c[i]));A=list(map(q,a[i]));Q=list(map(q,[*xy[i],1]))
        cc=[sum((abs(A[k])+abs(A[k+3]))*B[k] for k in range(3)),sum(abs(C[k]-A[k]-A[k+3])*B[k] for k in range(3)),sum(abs(C[k])*B[k] for k in range(3)),sum(abs(C[k]-A[k])*B[k] for k in range(3))]
        assert all(q(cache[i,j])>=v for j,v in enumerate(cc)),('cache',i)
        cq.append(C);aq.append(A);exact.append(sum(C[k]*Q[k] for k in range(3)));splitdot.append(sum((A[k]+A[k+3])*Q[k] for k in range(3)))
    # Symbolic rearrangement behind the enclosure, and exact constants at FP32 u.
    u=sp.Rational(1,2**24);gam=6*u/(1-6*u)
    assert gam**2<=49*u**2 and 1/(1-u)<=1+2*u
    U,R,G,D=sp.symbols('U R G D');assert sp.simplify((U*R+G+D)/(1-U)-(U*((U*R+G+D)/(1-U))+U*R+G+D))==0
    replay=set(np.flatnonzero(domain)[:1024].tolist())|{i for i,l in enumerate(labels) if l['family']=='control' and domain[i]}
    powerref=[mp.power(mp.mpf(float(u0)),mp.mpf(float(s0))) for u0,s0 in inp['power']]
    (OUT/'power_reference.json').write_text(json.dumps([mp.nstr(v,100) for v in powerref],indent=2)+'\n')
    arrays={};variant_results={};failures=[]
    for receipt in native['variants']:
        v=receipt['variant'];d=OUT/v
        for fname,key in [('probe','binary_sha256'),('edge.bin','edge_sha256'),('power.bin','power_sha256'),('sass.txt','sass_sha256')]:assert sha(d/fname)==receipt[key]
        raw=(d/'edge.bin').read_bytes();assert len(raw)==n*(34*4+8+4)
        out=np.frombuffer(raw[:n*136],'<f4').reshape(n,34);oracle=np.frombuffer(raw[n*136:n*144],'<f8');ng=np.frombuffer(raw[n*144:],'<u4');assert np.array_equal(ng.astype(bool),domain)
        arrays[v]=out;viol=Counter();eft=Counter();positive=Counter();errmax={'fast':0.,'compensated':0.};relative=[];outdomain=[];coarseRatios=[]
        for i in range(n):
            r,fast=q(out[i,1]),q(out[i,0]);P=exact[i];O=q(oracle[i]);Q=list(map(q,[*xy[i],1,*xy[i],1]))
            tests={'compensated_real':abs(r-P)<=q(out[i,2]),'compensated_native64':abs(r-O)<=q(out[i,3]),'coarse_real':abs(fast-P)<=q(out[i,4]),'coarse_native64':abs(fast-O)<=q(out[i,5])}
            if not domain[i]:
                outdomain.append(dict(index=i,label=labels[i],float_value=float(out[i,1]),native64=float(oracle[i]),violated_envelopes=[k for k,b in tests.items() if not b]));continue
            for k,b in tests.items():
                if not b:viol[k]+=1;failures.append(dict(variant=v,index=i,gate=k,label=labels[i]))
            for prefix,value,br,bo in [('coarse',fast,q(out[i,4]),q(out[i,5])),('compensated',r,q(out[i,2]),q(out[i,3]))]:
                if value>br:positive[prefix+'_real']+=1;assert P>0
                if value>bo:positive[prefix+'_native64']+=1;assert O>0
            for k in range(6):
                p0,e0=map(q,out[i,6+2*k:8+2*k]);assert p0+e0==aq[i][k]*Q[k],('product_EFT',v,i,k);eft['product']+=1
            p0=q(out[i,6])
            for k in range(1,6):
                h=q(out[i,6+2*k]);p1,q1=map(q,out[i,18+2*(k-1):20+2*(k-1)]);assert p1+q1==p0+h,('sum_EFT',v,i,k);p0=p1;eft['sum']+=1
            assert q(rn(p0+q(out[i,33])))==r,('final_RN',v,i)
            if i in replay:
                s=q(out[i,7]);assert s==q(out[i,28])
                for k in range(1,6):
                    qr=q(out[i,19+2*(k-1)])+q(out[i,7+2*k]);s=q(rn(s+q(rn(qr))));assert s==q(out[i,28+k]),('correction_RN',v,i,k)
            errmax['fast']=max(errmax['fast'],float(abs(fast-P)));errmax['compensated']=max(errmax['compensated'],float(abs(r-P)))
            if out[i,4]>0:coarseRatios.append(float(out[i,2]/out[i,4]))
            if P!=0:relative.append(float(abs(r-P)/abs(P)))
        raw=(d/'power.bin').read_bytes();assert len(raw)==m*16
        pf=np.frombuffer(raw[:m*8],'<f4').reshape(m,2);pd=np.frombuffer(raw[m*8:],'<f8');rows=[]
        for i,(u0,s0) in enumerate(inp['power']):
            ref=powerref[i];rr=np.float32(float(ref));candidates=[rr,np.nextafter(rr,np.float32(-np.inf)),np.nextafter(rr,np.float32(np.inf))]
            rounded=min(candidates,key=lambda t:(abs(mp.mpf(float(t))-ref),int(t.view(np.uint32))&1));bref=int(rounded.view(np.uint32));names=['named_powf','intrinsic_powf'];stat={}
            for j,key in enumerate(names):
                value=float(pf[i,j]);stat[key]=dict(value=value,abs_error=float(abs(mp.mpf(value)-ref)),relative_error=float(abs(mp.mpf(value)-ref)/ref),ulp_distance_from_RN=abs(int(pf[i,j].view(np.uint32))-bref))
            rows.append(dict(index=i,u=float(u0),sigma=float(s0),input_subnormal=bool(u0<2.**-126),reference_active=bool(mp.mpf(1)/255<=ref<=mp.mpf('.99')),metrics=stat,double_value=float(pd[i]),double_abs_error=float(abs(mp.mpf(float(pd[i]))-ref))))
        power_summary={}
        for subset,select in [('all',rows),('normal_input',[r for r in rows if not r['input_subnormal']]),('subnormal_input',[r for r in rows if r['input_subnormal']]),('active_normal_input',[r for r in rows if not r['input_subnormal'] and r['reference_active']])]:
            power_summary[subset]=dict(rows=len(select),**{key:dict(max_abs=max(r['metrics'][key]['abs_error'] for r in select),max_relative=max(r['metrics'][key]['relative_error'] for r in select),max_ulp_distance=max(r['metrics'][key]['ulp_distance_from_RN'] for r in select),above_documented_4ulp_guideline=sum(r['metrics'][key]['ulp_distance_from_RN']>4 for r in select)) for key in names})
        (d/'power_audit.json').write_text(json.dumps(dict(rows=rows,summary=power_summary,reference_dps=100,interval_proof=False,named_powf_is_standard=(v=='strict')),indent=2)+'\n')
        result=dict(in_domain=int(domain.sum()),rejected_domain=int((~domain).sum()),envelope_violations=dict(viol),EFT_identities=dict(eft),correction_replayed_rows=len(replay),certified_positive_counts=dict(positive),max_exact_plane_abs_errors=errmax,compensated_over_coarse_bound_quantiles=np.quantile(coarseRatios,[0,.5,.9,1]).tolist(),compensated_relative_error_quantiles=np.quantile(relative,[.5,.99,1]).tolist(),out_of_domain_controls=outdomain,power_summary=power_summary,all_edge_gates=not viol)
        variant_results[v]=result
        print(json.dumps(dict(variant=v,**{k:val for k,val in result.items() if k not in ['out_of_domain_controls','power_summary']})),flush=True)
    parity=bool(np.array_equal(arrays['strict'][domain],arrays['fast_ftz'][domain]));assert parity
    summary=dict(variants=variant_results,all_edge_gates=all(v['all_edge_gates'] for v in variant_results.values()),in_domain_strict_ftz_trace_bitwise_identical=parity,exact_cache_rows=n,used_source_checks=len(plan['used_sources']),symbolic_constants=dict(u=str(u),gamma6_squared=str(gam**2),upper=str(49*u**2),posterior_rearrangement=True),failures=failures,input_plan_sha256=sha(OUT/'input_plan.json'),audit_script_sha256=sha(Path(__file__)),sympy=sp.__version__,mpmath=mp.__version__,training=False,renderer=False,timing=False,power_interval_proved=False,performance_promotion=False,production_promotion=False)
    target.write_text(json.dumps(summary,indent=2)+'\n');assert summary['all_edge_gates'];print(json.dumps(dict(all_edge_gates=True,exact_cache_rows=n,used_sources=len(plan['used_sources']),strict_ftz_bitwise_parity=parity)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'audit_failure.txt').write_text(traceback.format_exc());raise
