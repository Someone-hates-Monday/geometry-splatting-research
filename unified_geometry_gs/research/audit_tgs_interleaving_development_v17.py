"""Exact symbolic and recorded-input development checks; not held-out validation."""
import itertools,json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import sympy as sp
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_interleaving_certificate_v17 import q,layers,certificate,endpoint_envelope
OLD=ROOT/'experiments/tgs_window_scene_v15/trace'
OUT=ROOT/'experiments/tgs_interleaving_v17'

def symbolic():
    a0,a1,b0,b1,p,c,d0,d1=sp.symbols('a0 a1 b0 b1 p c d0 d1',real=True)
    ext=[(b0,d0),(b1,d1)];children=[(a0,c),(a1,c)];R=(1-a0)*(1-a1);Q=1-R
    def comp(ls):
        T=sp.S.One;C=sp.S.Zero
        for a,color in ls:C+=T*a*color;T*=1-a
        return C
    checks=0
    for m in range(3):
        old=ext[:m]+[(p,c)]+ext[m:];Um=sp.prod(1-x[0] for x in ext[:m]);B=comp(ext[m:])
        for slots in itertools.combinations(range(4),2):
            actual=[];V=sp.S.One;U=sp.S.One;j=i=0;order=sp.S.Zero
            for k in range(4):
                if k in slots:
                    a,color=children[i];i+=1;actual.append((a,color));V*=1-a
                else:
                    b,d=ext[j];order+=U*b*(d-c)*(V-(1 if j<m else R));U*=1-b;j+=1;actual.append((b,d))
            assert sp.expand(comp(actual)-comp(old)-(Q-p)*Um*(c-B)-order)==0
            checks+=1
    return checks

def main():
    OUT.mkdir(exist_ok=True);plan_path=OUT/'development_plan.json'
    paths=[Path(__file__),ROOT/'tgs_interleaving_certificate_v17.py',ROOT/'TGS_INTERLEAVING_CERTIFICATE_NOTE_V17.md',
        OLD/'plan.json',OLD/'independent_audit.json',OLD/'endpoint_gate.json']
    paths+=sorted(OLD.glob('e*_actual.npz'));paths+=sorted(OLD.glob('e*_nostop.npz'))
    write_new(plan_path,dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},development_only=True,training=False,
        quality_claim=False,bound_ratio_is_descriptive=True))
    checks=symbolic();prior=json.loads((OLD/'plan.json').read_text());rows=[]
    for case in prior['pixels']:
        e=case['event'];x,y=case['pixel'];data={}
        for name in ['before','after']:
            full=np.load(OLD/f'e{e}_{x}_{y}_{name}_nostop.npz');raw=np.load(OLD/f'e{e}_{x}_{y}_{name}_actual.npz')
            alltrace=full['trace'];assert np.isfinite(alltrace[:,0]).all();trace=raw['trace'];trace=trace[np.isfinite(trace[:,0])]
            ls=layers(full['packed'],alltrace);env=endpoint_envelope(ls,trace)
            actual=[q(v) for v in raw['summary'][:3]]
            assert all(abs(a-b)<=r for a,b,r in zip(actual,env['prefix'],env['round_bound']))
            data[name]=dict(layers=ls,envelope=env,actual=actual)
        old,new=data['before'],data['after'];cert=certificate(old['layers'],new['layers'],case['parent_id'],2100000)
        bound=[v+old['envelope']['tail'][ch]+new['envelope']['tail'][ch]+old['envelope']['round_bound'][ch]+new['envelope']['round_bound'][ch]
            for ch,v in enumerate(cert['real_bound'])]
        delta=[a-b for a,b in zip(new['actual'],old['actual'])];assert all(abs(v)<=b for v,b in zip(delta,bound))
        peak=max(abs(v) for v in delta);bp=max(bound)
        rows.append(dict(event=e,pixel=[x,y],identity_exact=True,native_contained=True,
            actual_peak=float(peak),bound_peak=float(bp),bound_over_actual=float(bp/peak),
            window_bound=[float(x) for x in cert['window_bound']],order_bound=[float(x) for x in cert['order_bound']],
            active_crossed_external=cert['active_crossed_external'],early_stop=[old['envelope']['stop'],new['envelope']['stop']],
            accepted=[old['envelope']['accepted'],new['envelope']['accepted']],
            rounding_bound_sum=[float(a+b) for a,b in zip(old['envelope']['round_bound'],new['envelope']['round_bound'])]))
    # A software example of skip-current error much larger than nominal 1e-4.
    a=np.float32(.99);T=np.float32(1);first=np.float32(T*np.float32(1-a));test=np.float32(first*np.float32(1-a))
    assert first>=np.float32(1e-4) and test<np.float32(1e-4)
    missed=(1-q(a))*q(a);assert missed>q(np.float32(1e-4))*90
    write_new(OUT/'development_audit.json',dict(all_passed=True,symbolic_interleavings=checks,sympy=sp.__version__,rows=rows,
        synthetic_skip_current_tail=float(missed),not_independent_validation=True,training=False,novelty_established=False))
    print(json.dumps(dict(symbolic_checks=checks,rows=rows,synthetic_missed_tail=float(missed))),flush=True)

if __name__=='__main__':main()
