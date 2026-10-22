"""Independent effective-ray IFT oracle for the local backward refinement."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import sympy as sp

from research.median_solver import certified_bracket, log_fields


def quantities(t,a,p,w):
    delta=(t-p)/w
    g=a*np.exp(-.5*delta**2)
    v=1-g
    components=g*np.abs(delta)/(2*w*v)
    ht=float(components.sum())
    htt=float((np.sign(delta)*g*(v-delta**2)/(2*w*w*v*v)).sum())
    ha=np.where(t<=p,.5*np.exp(-.5*delta**2)/v,1/(1-a)-.5*np.exp(-.5*delta**2)/v)
    hp=-components
    hw=np.where(t<=p,1.,-1.)*.5*g*delta**2/(w*v)
    return np.exp(log_fields(t,a,p,w)[0]),ht,htt,np.r_[ha,hp,hw]


def refine(t,a,p,w):
    T,ht,htt,numerator=quantities(t,a,p,w)
    R=.5*ht
    if R<=1e-7 or T<=0:
        return None
    correction=(T-.5)/(T*ht)
    predicted=ht+htt*correction
    if not (np.isfinite(correction) and predicted>2e-7 and abs(correction)<.25*np.min(abs(t-p)) and abs(correction)<.0002):
        return None
    t1=t+correction
    return t1,-quantities(t1,a,p,w)[3]/predicted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error("Fresh output required.")
    a,r,z=sp.symbols("a r z",positive=True)
    g=a*sp.exp(-r*r*z*z/2)
    R=g*z*r*r/(4*(1-g))
    expected=g*r*r*((1-g)-r*r*z*z)/(4*(1-g)**2)
    assert sp.cancel(sp.diff(R,z)-expected)==0
    cases=[([.8,.65],[.8,1.2],[.03,.05]),
        ([.4,.7,.3],[.4,.5,.6],[.05,.08,.04]),
        ([.6,.55],[1.,1.05],[.03,.04])]
    records=[]
    for aa,pp,ww in cases:
        aa,pp,ww=(np.array(x) for x in (aa,pp,ww))
        lo,hi=certified_bracket(aa,pp,ww)
        root=brentq(lambda t:log_fields(t,aa,pp,ww)[0]+np.log(2),lo,hi,xtol=1e-14)
        _,ht,_,num=quantities(root,aa,pp,ww)
        oracle=-num/ht
        rows=[]
        for e in (1e-4,5e-5,2.5e-5,1.25e-5,6.25e-6):
            t0=root+e
            _,ht0,_,num0=quantities(t0,aa,pp,ww)
            result=refine(t0,aa,pp,ww)
            if result is None:
                raise ArithmeticError("Smooth test unexpectedly rejected refinement.")
            t1,gradient=result
            rows.append({"initial_root_error":e,"refined_root_error":abs(t1-root),
                "coarse_jacobian_relative_l2_error":float(np.linalg.norm(-num0/ht0-oracle)/np.linalg.norm(oracle)),
                "refined_jacobian_relative_l2_error":float(np.linalg.norm(gradient-oracle)/np.linalg.norm(oracle))})
        for j in range(1,len(rows)):
            rows[j]["jacobian_observed_order"]=float(np.log2(rows[j-1]["refined_jacobian_relative_l2_error"]/rows[j]["refined_jacobian_relative_l2_error"]))
        assert rows[-1]["jacobian_observed_order"]>1.9
        records.append({"alpha":aa.tolist(),"peaks":pp.tolist(),"widths":ww.tolist(),"root":root,"rows":rows})
    critical_rejected=refine(1e-4,np.array([.75]),np.array([0.]),np.array([.1])) is None
    assert critical_rejected
    result={"symbolic_denominator_derivative_pass":True,"smooth_cases":records,
        "critical_example_rejected":critical_rejected,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations":["Effective ray alpha/peak/width Jacobians, not full Gaussian world parameters.",
            "Smooth separated-peak tests verify the local order, not a uniform critical-point bound.",
            "The forward loss still uses its coarse root, so its upstream derivative is not quadratically corrected.",
            "Float64 oracle; CUDA float32 and full rendering require separate validation."]}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"smooth_cases":len(records),"critical_example_rejected":critical_rejected,
        "min_final_jacobian_order":min(c["rows"][-1]["jacobian_observed_order"] for c in records)},indent=2))


if __name__=="__main__":
    main()
