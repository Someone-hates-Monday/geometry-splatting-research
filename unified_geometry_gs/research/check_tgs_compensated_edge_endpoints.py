"""Post-output descriptive checks, no new GPU calls or adaptive input changes."""
import hashlib,json,math,struct
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compensated_edge_v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(x):return F.from_float(float(x))
def main():
    assert not (OUT/'endpoint_checks.json').exists()
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_edge_gates']
    plan=json.loads((OUT/'input_plan.json').read_text());labels=json.loads((OUT/'labels.json').read_text())
    with np.load(OUT/'inputs.npz') as z:c=z['coeff'];xy=z['xy'];domain=z['gate'];power=z['power']
    n=len(c);raw=(OUT/'strict/edge.bin').read_bytes();out=np.frombuffer(raw[:n*136],'<f4').reshape(n,34);oracle=np.frombuffer(raw[n*136:n*144],'<f8')
    family={};worst=None;correct=0;sign=Counter()
    for i in np.flatnonzero(domain):
        # CPython evaluates separate binary64 operations, not fused expressions.
        p=float(c[i,0])*float(xy[i,0]);t=float(c[i,1])*float(xy[i,1]);s=p+t;v=s+float(c[i,2])
        assert struct.pack('<d',v)==struct.pack('<d',oracle[i]),('native64_endpoint',int(i));correct+=1
        P=sum(q(vv)*q(ww) for vv,ww in zip(c[i],[*xy[i],1]));r=q(out[i,1]);f=q(out[i,0]);ae=float(abs(P-r));fe=float(abs(P-f));rel=float(abs(r-P)/abs(P)) if P else None
        key=labels[i]['family']+'/'+labels[i].get('point',labels[i].get('kind','unknown'))
        family.setdefault(key,[]).append([ae,fe,float(out[i,2]),float(out[i,4]),abs(float(P))])
        sgn=lambda x:(x>0)-(x<0)
        sign['naive_wrong_sign']+=sgn(P)!=sgn(f);sign['compensated_wrong_sign']+=sgn(P)!=sgn(r)
        if rel is not None and (worst is None or rel>worst['relative_error']):worst=dict(index=int(i),label=labels[i],coeff=c[i].tolist(),xy=xy[i].tolist(),exact_plane_rational=str(P),exact_plane=float(P),compensated=float(out[i,1]),coarse=float(out[i,0]),absolute_error=ae,relative_error=rel,compensated_bound=float(out[i,2]),positive_certificate=bool(out[i,1]>out[i,2]),negative_certificate=bool(-out[i,1]>out[i,2]))
    descriptive={key:dict(rows=len(vals),absolute_error_quantiles=np.quantile(np.array(vals)[:,:2],[.5,.99,1],axis=0).tolist(),compensated_over_coarse_bound_median=float(np.median(np.array(vals)[:,2]/np.array(vals)[:,3])),near_zero_reference_rows=sum(v[4]<1e-6 for v in vals)) for key,vals in family.items()}
    # Repeat all finite-grid reference calculations at 200 digits. Check exact RN32 endpoints
    # and compare the saved 100-digit strings at a 90-digit relative precision threshold.
    old=json.loads((OUT/'power_reference.json').read_text());refchecks=0;maxdelta=mp.mpf(0)
    with mp.workdps(200):
        for (u,s),text in zip(power,old):
            new=mp.power(mp.mpf(float(u)),mp.mpf(float(s)));delta=abs(new-mp.mpf(text))/new;maxdelta=max(maxdelta,delta);assert delta<mp.mpf('1e-90');refchecks+=1
    # Static SASS evidence, counted only within edge entry; not dynamic instruction cost.
    sass={}
    for variant in ['strict','fast_ftz']:
        text=(OUT/variant/'sass.txt').read_text();start=text.index('Function : _Z4edge');part=text[start:];nextfn=part.find('Function :',20)
        if nextfn>=0:part=part[:nextfn]
        sass[variant]=dict(float_FTZ_markers=part.count('.FTZ'),FFMA_lines=sum('FFMA' in l for l in part.splitlines()),FADD_lines=sum('FADD' in l for l in part.splitlines()),file_sha256=sha(OUT/variant/'sass.txt'))
    result=dict(native64_CPU_bitwise_endpoints=correct,descriptive_groups=descriptive,raw_sign_disagreements=dict(sign),worst_compensated_relative_error=worst,power_200dps_rechecks=refchecks,max_100dps_relative_discrepancy=mp.nstr(maxdelta,12),power_interval_proof=False,SASS_static_edge=sass,audit_sha256=sha(OUT/'independent_audit.json'),script_sha256=sha(Path(__file__)),timing=False,training=False)
    (OUT/'endpoint_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='descriptive_groups'}),flush=True)
if __name__=='__main__':main()
