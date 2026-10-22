"""Exact interval lemma checks on preserved v7 samples, no GPU or model changes."""
import hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import sympy as sp
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_real_work_v7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(x):return F.from_float(float(x))
def main():
    assert not (OUT/'active_interval_checks.json').exists();audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_gates']
    r=[-F(1,2**20),-F(512),-F(256)];e=[F(1,2**40),F(1,2**15),F(1,2**16)];l=[a-b for a,b in zip(r,e)];u=[a+b for a,b in zip(r,e)];assert l[0]>max(u[1:]);assert max(u)<0;assert abs(max(r))<=max(e);assert max(e)/e[0]==2**25
    # Symbolic sensitivity is monotone in the uncertainty for fixed positive lower denominator.
    d,U,S=sp.symbols('d U S',positive=True);deriv=sp.factor(sp.diff(S*d/(U-d),d));assert sp.simplify(deriv-S*U/(U-d)**2)==0
    rows=[]
    for ci in range(5):
        with np.load(OUT/f'camera_{ci}_raw.npz') as z:s=z['samples']
        ratios=[];count=0
        for v in s:
            if v[0]==0 or int(v[5])!=3:continue
            rr=list(map(q,v[[10,12,14]]));ee=list(map(q,v[[11,13,15]]));a=int(v[22]);ll=[x-y for x,y in zip(rr,ee)];uu=[x+y for x,y in zip(rr,ee)]
            assert ll[a]>max(uu[k] for k in range(3) if k!=a);assert max(ll)==ll[a] and max(uu)==uu[a];assert max(rr)==rr[a];assert max(uu)<0
            assert max(ee)==q(v[9]);ratios.append(float(max(ee)/ee[a]));count+=1
        rows.append(dict(camera=ci,name=audit['rows'][ci]['name'],inside_unique_samples=count,max_radius_over_active_radius_quantiles=np.quantile(ratios,[0,.5,.9,.99,1]).tolist()))
    result=dict(counterexample=dict(r=list(map(str,r)),e=list(map(str,e)),old_sign_inconclusive=True,tight_inside_unique=True,radius_ratio=2**25),sample_rows=rows,checked_inside_unique=sum(r['inside_unique_samples'] for r in rows),sensitivity_derivative=str(deriv),sympy=sp.__version__,note_sha256=sha(ROOT/'TGS_ACTIVE_EDGE_INTERVAL_NOTE.md'),audit_sha256=sha(OUT/'independent_audit.json'),script_sha256=sha(Path(__file__)),implementation=False,training=False,timing=False,performance_claim=False)
    (OUT/'active_interval_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
