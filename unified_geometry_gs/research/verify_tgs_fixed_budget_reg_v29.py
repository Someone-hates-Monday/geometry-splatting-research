"""Exact binary32-data ratio identities and observed GPU FP32 mean coefficients."""
import json,traceback
from fractions import Fraction
from pathlib import Path
import numpy as np
import sympy as sp
import torch
from tgs_joint_policy_data_v29 import ROOT,OUT,sha,write
from tgs_cross_scene_common_v10 import CrossWorkload
def exact_sum(x):
    assert x.dtype==np.float32 and np.isfinite(x).all() and np.all(x>=0) and np.all(x<=1)
    bits=x.reshape(-1).view(np.uint32);exp=((bits>>23)&255).astype(np.int64);mant=(bits&0x7fffff).astype(np.uint64);mant[exp>0]+=1<<23
    assert len(x.reshape(-1))*(1<<24)<(1<<53)
    groups=np.bincount(exp,weights=mant.astype(np.float64),minlength=128);integer=int(groups[0])
    for e in range(1,len(groups)):integer+=int(groups[e])<<(e-1)
    return Fraction(integer,1<<149)
@torch.no_grad()
def main():
    choices=json.loads((OUT/'choices.json').read_text());deps=[Path(__file__),ROOT/'TGS_FIXED_BUDGET_REG_NOTE_V29.md',OUT/'choices.json',OUT/'execution_plan.json',ROOT/'tgs_cross_scene_common_v10.py']
    write(OUT/'global_reg_plan.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in deps},supplement_after_calibration_selection=True,choices_changed=False,new_native_forwards=0,training=False,exact_effective_binary32_values=True,lambda_not_assumed=True))
    n,k,s,o,lam=sp.symbols('N k S O lambda',positive=True);direct=lam*((s+3*o)/(n+3*k)-s/n);fixed=3*lam*(o-k*s/n)/(n+3*k);one=3*lam*(o-k*s/n)/(n+3)
    assert sp.simplify(direct-fixed)==0 and sp.simplify(direct-one+9*lam*(k-1)*(o-k*s/n)/((n+3*k)*(n+3)))==0
    scenes=[]
    for sc in choices['scenes']:
        w=CrossWorkload(sc['scene']);initial=w.input_hashes();assert initial==sc['model_inputs'];x=w.opacity.detach().cpu().numpy().reshape(-1);N=len(x);S=exact_sum(x);mu=S/N;base=w.opacity.abs().mean();base_value=float(base.item());rows=[]
        for name,parents in sc['selected'].items():
            K=len(parents);assert K==8;O=sum((Fraction.from_float(float(x[p])) for p in parents),Fraction(0));delta=(S+3*O)/(N+3*K)-mu;weights=[3*(Fraction.from_float(float(x[p]))-mu)/(N+3*K) for p in parents];assert sum(weights,Fraction(0))==delta
            wrong=3*(O-K*mu)/(N+3);eta=-9*(K-1)*(O-K*mu)/((N+3*K)*(N+3));assert delta-wrong==eta
            keep=torch.ones(N,device='cuda',dtype=torch.bool);keep[parents]=False;joint=torch.cat([w.opacity[keep],*[w.opacity[p:p+1].repeat((4,)+(1,)*(w.opacity.ndim-1)) for p in sorted(parents)]])
            observed=float(joint.abs().mean().item())-base_value
            rows.append(dict(policy=name,parents=parents,exact_joint_coefficient=str(delta),joint_coefficient=float(delta),fixed_budget_unary_sum_exact=True,sum_true_single_coefficients=float(wrong),single_sum_vs_joint_difference=float(eta),observed_FP32_mean_delta=observed,FP32_delta_minus_exact=float(observed-float(delta)),per_candidate_fixed_budget_coefficients=[float(v) for v in weights]))
            del joint,keep
        assert initial==w.input_hashes();scenes.append(dict(scene=sc['scene'],N=N,S_exact=str(S),mean_opacity_exact=str(mu),mean_opacity_float64=float(mu),observed_base_FP32_mean=base_value,effective_opacity_sha256=initial['opacity'],rows=rows,model_inputs_unchanged=True));del w,x;torch.cuda.empty_cache()
    write(OUT/'global_reg_results.json',dict(all_identities_passed=True,scenes=scenes,sympy=sp.__version__,torch=torch.__version__,plan_sha256=sha(OUT/'global_reg_plan.json'),new_native_forwards=0,training=False,lambda_unknown=True,full_training_objective_measured=False,choices_unchanged=True))
    print(json.dumps(dict(stage='REAL_INPUT_FIXED_BUDGET_REG_IDENTITIES_PASSED',scenes=[dict(scene=s['scene'],N=s['N'],primary=s['rows'][0]) for s in scenes])),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'global_reg_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
