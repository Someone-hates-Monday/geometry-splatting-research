"""Exact recorded-alpha identity, native rounding model and bound coverage on frozen pixels."""
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new,exact
from tgs_interleaving_certificate_v17 import q,layers,certificate,endpoint_envelope
from audit_tgs_compensated_depth_trace import rn
OUT=ROOT/'experiments/tgs_interleaving_v17/validation'

def audit_endpoint(path):
    a=np.load(path);packet=a['packed'];full=a['full'];trace=a['actual'];trace=trace[np.isfinite(trace[:,0])];summary=a['summary']
    assert len(packet)==len(full) and np.isfinite(full[:,[0,1,4,5,6,7,8,9,10,11,12,13]]).all()
    assert np.isfinite(packet).all() and np.all(packet[:,13:16]>=0)
    assert np.array_equal(packet[:,16],full[:,0])
    assert np.array_equal(trace[:,[0,1,4,5]],full[:len(trace)][:,[0,1,4,5]])
    assert exact(summary[:5],a['native'].astype(np.float32))
    T=F(1);C=[F(0)]*3;last=0;checks=0;fma=sep=0
    threshold=q(np.float32(1e-4));cut=q(np.float32(1/255));stopped=False
    for i,r in enumerate(trace):
        assert q(r[6])==T
        al=q(r[4]);eligible=not bool(r[5]) and al>=cut
        assert F(0)<=al<=F(1)
        if eligible:
            test=q(rn(T*q(rn(1-al))));assert test==q(r[7])
            stopped=test<threshold;assert bool(r[9])==stopped and bool(r[8])==(not stopped)
            if not stopped:
                for ch in range(3):
                    first=q(rn(q(packet[i,13+ch])*al));fused=q(rn(first*T+C[ch]));separate=q(rn(q(rn(first*T))+C[ch]));value=q(r[11+ch])
                    assert value in [fused,separate],(path,i,ch)
                    fma+=value==fused;sep+=value==separate;C[ch]=value
                T=test;last=i+1;checks+=1
        else:assert not r[8] and not r[9]
        assert q(r[10])==T and all(q(r[11+ch])==C[ch] for ch in range(3))
        if stopped:assert i==len(trace)-1
    assert q(rn(1-T))==q(summary[3]) and int(summary[4])==last
    ls=layers(packet,full);env=endpoint_envelope(ls,trace)
    native=[q(v) for v in summary[:3]]
    assert all(abs(a-b)<=e for a,b,e in zip(native,env['prefix'],env['round_bound']))
    return dict(layers=ls,envelope=env,native=native,parent_color=[q(x) for x in a['parent_color']],checks=checks,fma=fma,separate=sep)

def main():
    plan=json.loads((OUT/'plan.json').read_text());results=json.loads((OUT/'results.json').read_text())
    assert results['full_forwards']==12 and results['trace_calls']==256 and results['all_endpoints_literal']
    assert results['plan_sha256']==sha(OUT/'plan.json')
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    write_new(OUT/'audit_plan_v2.json',dict(script_sha256=sha(Path(__file__)),RN_helper_sha256=sha(ROOT/'audit_tgs_compensated_depth_trace.py'),
        results_sha256=sha(OUT/'results.json'),analysis='Exact rational containment; no threshold fitting',training=False))
    rows=[];RN=FMA=SEP=0
    for row in results['rows']:
        old=audit_endpoint(OUT/row['before']);new=audit_endpoint(OUT/row['after']);RN+=old['checks']+new['checks'];FMA+=old['fma']+new['fma'];SEP+=old['separate']+new['separate']
        assert old['parent_color']==new['parent_color']
        before=old['layers'];dummy=not any(x[0]==row['parent_id'] for x in before)
        if dummy:before=[(row['parent_id'],F(0),old['parent_color'])]+before
        cert=certificate(before,new['layers'],row['parent_id'],row['N'])
        bound=[b+old['envelope']['tail'][ch]+new['envelope']['tail'][ch]+old['envelope']['round_bound'][ch]+new['envelope']['round_bound'][ch]
            for ch,b in enumerate(cert['real_bound'])]
        delta=[a-b for a,b in zip(new['native'],old['native'])];assert all(abs(v)<=b for v,b in zip(delta,bound)),row
        # Also check the exact native decomposition, without fitting a residual.
        real_delta=cert['actual']
        for ch in range(3):
            errnew=new['native'][ch]-new['envelope']['prefix'][ch];errold=old['native'][ch]-old['envelope']['prefix'][ch]
            assert delta[ch]==real_delta[ch]-new['envelope']['tail'][ch]+old['envelope']['tail'][ch]+errnew-errold
        peak=max(abs(x) for x in delta);bp=max(bound)
        rows.append(dict(**row,identity_exact=True,native_contained=True,normal_range_premise=True,zero_parent_inserted=dummy,
            actual_peak=float(peak),bound_peak=float(bp),bound_over_actual=float(bp/peak) if peak else None,
            active_crossed_external=cert['active_crossed_external'],early_stop=[old['envelope']['stop'],new['envelope']['stop']],
            tail_bound=[float(a+b) for a,b in zip(old['envelope']['tail'],new['envelope']['tail'])],
            window_bound=[float(x) for x in cert['window_bound']],order_bound=[float(x) for x in cert['order_bound']],
            rounding_bound_sum=[float(a+b) for a,b in zip(old['envelope']['round_bound'],new['envelope']['round_bound'])]))
        print(json.dumps(dict(done=len(rows),of=64)),flush=True) if len(rows)%16==0 else None
    ratios=[r['bound_over_actual'] for r in rows if r['bound_over_actual'] is not None]
    stopped=[r for r in rows if any(r['early_stop'])]
    aggregate=dict(pixels=len(rows),nonzero_error=len(ratios),zero_error=len(rows)-len(ratios),
        minimum_ratio=min(ratios) if ratios else None,median_ratio=float(np.median(ratios)) if ratios else None,
        maximum_ratio=max(ratios) if ratios else None,early_stop_pairs=len(stopped),
        crossed_pairs=sum(r['active_crossed_external']>0 for r in rows),zero_parent_insertions=sum(r['zero_parent_inserted'] for r in rows),
        maximum_bound=max(r['bound_peak'] for r in rows),transmittance_RN_checks=RN,RGB_fma_compatible=FMA,RGB_separate_compatible=SEP)
    write_new(OUT/'independent_audit_v2.json',dict(all_passed=True,aggregate=aggregate,rows=rows,training=False,
        compiler_semantics_fully_certified=False,independent_photographic_confirmation=False,novelty_established=False))
    print(json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
