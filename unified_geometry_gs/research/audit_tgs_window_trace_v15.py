"""Exact recorded-alpha RGB decomposition; no power or full-renderer correctness theorem."""
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new,exact
from audit_tgs_compensated_depth_trace import rn
OUT=ROOT/'experiments/tgs_window_scene_v15/trace'

def q(x):return F.from_float(float(x))

def real_composite(packet,trace):
    T=F(1);C=[F(0)]*3;checks=0
    for p,r in zip(packet,trace):
        assert int(p[16])==int(r[0]) and not bool(r[9])
        alpha=q(r[4]);expected=(not bool(r[5])) and alpha>=q(np.float32(1/255))
        assert bool(r[8])==expected
        if expected:
            for ch in range(3):C[ch]+=q(p[13+ch])*alpha*T
            T*=1-alpha
        checks+=1
    return C,T,checks

def audit_T(trace,summary):
    T=F(1);last=0;checks=0
    for r in trace:
        assert q(r[6])==T
        if r[8]:
            test=q(rn(T*q(rn(1-q(r[4])))))
            assert test==q(r[7]);T=test;last=int(r[1]);checks+=1
        assert q(r[10])==T
    assert q(rn(1-T))==q(summary[3]) and int(summary[4])==last and summary[5]==0
    return checks

def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    result=json.loads((OUT/'results.json').read_text());plan=json.loads((OUT/'plan.json').read_text());gate=json.loads((OUT/'endpoint_gate.json').read_text())
    assert result['all_primary_endpoints_literal'] and result['all_native_fields_literal'] and result['full_forwards']==3 and result['trace_calls']==18
    assert gate['all_literal'] and gate['endpoints']==6
    assert result['plan_sha256']==sha(OUT/'plan.json') and result['script_sha256']==sha(ROOT/'trace_tgs_window_scene_v15.py')
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h
    build=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(build['path']))==build['sha256']
    assert build['source_sha256']==sha(ROOT/'tgs_window_scene_trace_v15.cu')
    outputs=[];RNchecks=0;packetchecks=0
    for pixel in plan['pixels']:
        e=pixel['event'];x,y=pixel['pixel'];data={}
        for mode in ['before_actual','after_actual','before_nostop','after_nostop','after_group','after_group_nostop']:
            f=OUT/f'e{e}_{x}_{y}_{mode}.npz';raw=np.load(f)
            trace=raw['trace'];trace=trace[np.isfinite(trace[:,0])];packet=raw['packed'];summary=raw['summary']
            assert len(trace)==len(packet)
            recorded=next(r for r in result['rows'] if r['event']==e and r['pixel']==[x,y] and r['mode']==mode)
            assert np.array_equal(summary,np.array(recorded['summary'],dtype=np.float32))
            RNchecks+=audit_T(trace,summary);real,T,checks=real_composite(packet,trace);packetchecks+=checks
            data[mode]=dict(packet=packet,trace=trace,summary=summary,real=real,T=T)
        for mode,kind in [('before_actual','before'),('after_actual',f'event_{e}')]:
            native=np.load(OUT/f'{kind}_{x}_{y}_packet.npz')
            assert exact(data[mode]['summary'][:5],native['native'].astype(np.float32))
            assert exact(data[mode]['packet'],native['packed'])
        assert exact(data['before_actual']['summary'],data['before_nostop']['summary'])
        assert exact(data['after_actual']['summary'],data['after_nostop']['summary'])
        assert exact(data['after_group']['summary'],data['after_group_nostop']['summary'])
        before,after,group=[data[k] for k in ['before_actual','after_actual','after_group']]
        oldids=before['packet'][:,16].astype(np.int64);newids=after['packet'][:,16].astype(np.int64)
        N=2100000;child=newids>=N-1;parent=pixel['parent_id']
        canonical=np.where(newids<parent,newids,newids+1)
        assert np.array_equal(oldids[oldids!=parent],canonical[~child])
        assert exact(before['packet'][oldids!=parent,:16],after['packet'][~child,:16])
        pos=int(np.flatnonzero(oldids==parent)[0]);external=after['packet'][~child]
        expected=np.concatenate([external[:pos],after['packet'][child],external[pos:]])
        assert exact(expected,group['packet'])
        assert np.all(after['packet'][child,13:16]==before['packet'][oldids==parent,13:16])
        alpha_after={int(r[0]):q(r[4]) for r in after['trace']}
        assert all(q(r[4])==alpha_after[int(r[0])] for r in group['trace'])
        terms=[]
        for ch in range(3):
            actual=q(after['summary'][ch])-q(before['summary'][ch])
            window=group['real'][ch]-before['real'][ch]
            order=after['real'][ch]-group['real'][ch]
            rounding=(q(after['summary'][ch])-after['real'][ch])-(q(before['summary'][ch])-before['real'][ch])
            assert window+order+rounding==actual
            terms.append(dict(actual=str(actual),window_fixed_group=str(window),ordering=str(order),rounding=str(rounding),
                              floats=dict(actual=float(actual),window_fixed_group=float(window),ordering=float(order),rounding=float(rounding))))
        actual_peak=float(np.max(np.abs(after['summary'][:3].astype(float)-before['summary'][:3])))
        group_peak=float(np.max(np.abs(group['summary'][:3].astype(float)-before['summary'][:3])))
        outputs.append(dict(event=e,pixel=[x,y],all_native_endpoints_literal=True,no_early_stop_any_mode=True,
            all_external_packets_literal_identical=True,all_child_alphas_unchanged_by_reorder=True,
            native_perturbation_peak=actual_peak,fixed_group_perturbation_peak=group_peak,
            fixed_group_over_native_peak=group_peak/actual_peak,
            exact_three_channel_decomposition=terms,
            maximum_rounding_residual=max(abs(t['floats']['rounding']) for t in terms)))
    write_new(target,dict(all_verified=True,rows=outputs,endpoint_count=6,trace_calls=18,
        exact_recorded_alpha_decompositions=True,transmittance_RN_checks=RNchecks,packet_checks=packetchecks,
        script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'results.json'),plan_sha256=sha(OUT/'plan.json'),
        RN_helper_sha256=sha(ROOT/'audit_tgs_compensated_depth_trace.py'),
        sorting_mechanism_supported_on_three_frozen_pixels=True,all_image_cause_claim=False,novelty_established=False,training=False))
    print(json.dumps(dict(all_verified=True,rows=[{k:v for k,v in r.items() if k!='exact_three_channel_decomposition'} for r in outputs],RN_checks=RNchecks)),flush=True)

if __name__=='__main__':main()
