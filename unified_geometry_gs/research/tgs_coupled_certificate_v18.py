"""Conditional coupled bounds on recorded CUDA inputs, exact rational arithmetic."""
from fractions import Fraction as F
import numpy as np
from tgs_interleaving_certificate_v17 import q,compose,certificate
from audit_tgs_interleaving_validation_v17_v2 import audit_endpoint

def accepted_data(path):
    a=np.load(path);trace=a['actual'];trace=trace[np.isfinite(trace[:,0])]
    accepted=[];states=[]
    for i,r in enumerate(trace):
        if r[8]:
            accepted.append((int(a['packed'][i,16]),q(r[4]),[q(x) for x in a['packed'][i,13:16]]))
            states.append(([q(x) for x in r[11:14]],q(r[10])))
    return accepted,states

def coupled(before_path,after_path,parent,N):
    old=audit_endpoint(before_path);new=audit_endpoint(after_path)
    assert old['parent_color']==new['parent_color']
    before=old['layers'];dummy=not any(x[0]==parent for x in before)
    if dummy:before=[(parent,F(0),old['parent_color'])]+before
    cert=certificate(before,new['layers'],parent,N)
    eo,en=old['envelope'],new['envelope'];H=cert['real_bound']
    baseline=[H[ch]+eo['tail'][ch]+en['tail'][ch]+eo['round_bound'][ch]+en['round_bound'][ch] for ch in range(3)]
    signed=[H[ch]+abs(eo['tail'][ch]-en['tail'][ch])+eo['round_bound'][ch]+en['round_bound'][ch] for ch in range(3)]
    a,sa=accepted_data(before_path);b,sb=accepted_data(after_path);k=0
    while k<min(len(a),len(b)) and a[k][1:]==b[k][1:]:k+=1
    if k:
        assert sa[k-1]==sb[k-1]
        C0,T0=sa[k-1]
    else:C0=[F(0)]*3;T0=F(1)
    t0=F(1)
    for _,al,_ in a[:k]:t0*=1-al
    assert t0>0;scale=T0/t0
    u=F(1,2**24)
    def local(ls):
        l=len(ls);assert 3*l*u<1
        gamma=3*l*u/(1-3*l*u);S=compose(ls)
        return [gamma*(C0[ch]+T0*S[ch]) for ch in range(3)]
    ra,rb=local(a[k:]),local(b[k:])
    prefix=[scale*(H[ch]+abs(eo['tail'][ch]-en['tail'][ch]))+ra[ch]+rb[ch] for ch in range(3)]
    identical=k==len(a)==len(b)
    improved=[F(0)]*3 if identical else [min(s,p) for s,p in zip(signed,prefix)]
    delta=[x-y for x,y in zip(new['native'],old['native'])]
    for name,bound in [('baseline',baseline),('signed',signed),('prefix',prefix),('improved',improved)]:
        assert all(abs(v)<=lim for v,lim in zip(delta,bound)),name
    assert all(i<=s<=v for i,s,v in zip(improved,signed,baseline))
    peak=max(abs(x) for x in delta)
    result=dict(actual_peak=float(peak),native_contained=True,zero_parent_inserted=dummy,
        early_stop=[eo['stop'],en['stop']],common_accepted_prefix=k,accepted=[len(a),len(b)],
        divergent_suffix=[len(a)-k,len(b)-k],full_candidate_records=[len(old['layers']),len(new['layers'])],
        exact_zero_by_input_equality=identical,active_crossed_external=cert['active_crossed_external'])
    for name,bound in [('v17',baseline),('signed_tail',signed),('coupled',improved)]:
        value=max(bound);result[name]=dict(bound_peak=float(value),ratio=float(value/peak) if peak else None,
            channel_bounds=[float(x) for x in bound])
    return result

def aggregate(rows):
    result=dict(pixels=len(rows),nonzero=sum(r['actual_peak']>0 for r in rows),zero=sum(r['actual_peak']==0 for r in rows),
        early_stop_pairs=sum(any(r['early_stop']) for r in rows),exact_zero_certificates=sum(r['exact_zero_by_input_equality'] for r in rows),
        candidate_records=sum(sum(r['full_candidate_records']) for r in rows),accepted_records=sum(sum(r['accepted']) for r in rows),
        divergent_suffix_records=sum(sum(r['divergent_suffix']) for r in rows))
    for name in ['v17','signed_tail','coupled']:
        ratios=[r[name]['ratio'] for r in rows if r[name]['ratio'] is not None]
        result[name]=dict(min_ratio=min(ratios) if ratios else None,median_ratio=float(np.median(ratios)) if ratios else None,
            max_ratio=max(ratios) if ratios else None,max_bound=max(r[name]['bound_peak'] for r in rows),
            zero_error_positive_bound=sum(r['actual_peak']==0 and r[name]['bound_peak']>0 for r in rows))
    return result
