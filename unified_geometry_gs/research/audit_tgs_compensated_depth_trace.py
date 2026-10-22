"""Exact alpha/transmittance/weight/FMA attribution for all frozen failures."""
import hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compensated_renderer_v8';D=OUT/'depth_trace'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def q(v):return F.from_float(float(v))
def rn(x):
    v=np.float32(float(x));choices=[v,np.nextafter(v,np.float32(-np.inf)),np.nextafter(v,np.float32(np.inf))];return min(choices,key=lambda a:(abs(q(a)-x),int(a.view(np.uint32))&1))
def real_depth(tr):
    T=F(1);d=F(0)
    for r in tr:
        if r[11]:d+=q(r[14])*q(r[5])*T;T*=1-q(r[5])
    return d
def weight_sum(tr):return sum(q(r[14])*q(r[20]) for r in tr if r[11])
def audit_rn(tr,summary):
    T=F(1);D=F(0);checks=0
    for r in tr:
        assert T==q(r[9]);a=q(r[5]);w=q(rn(a*T));assert w==q(r[20]);checks+=1
        if not r[6] and a>=q(np.float32(1/255)):
            test=q(rn(T*q(rn(1-a))));assert test==q(r[10]);checks+=1
            if r[11]:D=q(rn(q(r[14])*w+D));T=test
        assert D==q(r[18]) and T==q(r[13]);checks+=2
    assert D==q(summary[3]);return checks
def main():
    target=D/'independent_audit.json';assert not target.exists();plan=json.loads((D/'plan.json').read_text());results=json.loads((D/'results.json').read_text());assert results['all_endpoints_bitwise'];assert results['plan_sha256']==sha(D/'plan.json');assert plan['protocol_sha256']==sha(ROOT/'TGS_COMPENSATED_DEPTH_TRACE_PROTOCOL.md');assert plan['script_sha256']==sha(ROOT/'trace_tgs_compensated_depth.py')
    modes=plan['modes'];rows=[];calls=0;summary={name:dict(depth_bad=0,depth_bitwise=0,max_depth_abs=0.,support_changed=0,max_RGB_abs=0.) for name in modes}
    for pi,rr in enumerate(results['rows']):
        data={}
        for name in modes:
            with np.load(D/f'pixel_{pi}_{name}.npz') as z:tr=z['trace'];tr=tr[np.isfinite(tr[:,0])];ss=z['summary'];data[name]=(tr,ss)
        with np.load(D/f'packet_{pi}.npz') as z:assert np.array_equal(data['guard'][1][[0,1,2,3,4,5,6,8]],z['guard_endpoint']);assert np.array_equal(data['oracle'][1][[0,1,2,3,4,5,6,8]],z['oracle_endpoint'])
        g,gs=data['guard'];o,os=data['oracle'];assert np.array_equal(g[:,[0,1,11,12]],o[:,[0,1,11,12]])
        calls+=audit_rn(g,gs)+audit_rn(o,os);G,O=real_depth(g),real_depth(o);WG,WO=weight_sum(g),weight_sum(o);DG,DO=q(gs[3]),q(os[3]);terms=[G-O,(WG-G)-(WO-O),(DG-WG)-(DO-WO)];assert sum(terms)==DG-DO
        alpha_diff=np.abs(g[:,5]-o[:,5]);include=g[:,11]>0;floatweightdiff=np.abs(g[:,20]-o[:,20]);alpha_same=int(np.count_nonzero(alpha_diff==0))
        row=dict(index=pi,pixel=rr['pixel'],evaluated=len(g),included=int(include.sum()),support_identical=True,stop_guard=int(gs[7]),stop_oracle=int(os[7]),observed_depth_difference=float(DG-DO),exact_terms_rational=list(map(str,terms)),exact_terms_float=dict(alpha_real=float(terms[0]),T_and_weight_rounding=float(terms[1]),depth_FMA_rounding=float(terms[2])),alpha_max_abs=float(alpha_diff.max()),alpha_same_count=alpha_same,max_float_weight_difference=float(floatweightdiff.max()),guard_rounding_residual_vs_recorded_alpha=float(DG-G),oracle_rounding_residual_vs_recorded_alpha=float(DO-O))
        rows.append(row)
        for name,(tr,ss) in data.items():
            delta=float(ss[3]-os[3]);s=summary[name];s['depth_bad']+=abs(delta)>3e-5;s['depth_bitwise']+=ss[3]==os[3];s['max_depth_abs']=max(s['max_depth_abs'],abs(delta));s['support_changed']+=not np.array_equal(tr[:,[0,1,11,12]],o[:,[0,1,11,12]]);s['max_RGB_abs']=max(s['max_RGB_abs'],float(np.abs(ss[:3]-os[:3]).max()))
    for v in summary.values():
        for k,x in v.items():v[k]=int(x) if k in ['depth_bad','depth_bitwise','support_changed'] else float(x)
    result=dict(rows=rows,mode_summary=summary,all_exact_decompositions=True,RN_checks=calls,all_native_endpoints_bitwise=True,early_stop_on_failure_pixels=sum(r['stop_guard']>0 or r['stop_oracle']>0 for r in rows),real_renderer_gate_still_false=True,power_certificate_proved=False,performance_measurement_allowed=False,script_sha256=sha(Path(__file__)),results_sha256=sha(D/'results.json'),training=False)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
