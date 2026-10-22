"""Exact branch derivatives and float64 oracle for the Halley adjoint prototype."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import sympy as sp
from research.median_solver import certified_bracket, log_fields
from research.verify_adjoint_refinement import quantities


def refine(t,a,p,w):
    T,ht,htt,num=quantities(t,a,p,w)
    d=(t-p)/w
    g=a*np.exp(-.5*d*d)
    v=1-g
    httt=float((.5*g*np.abs(d)/w**3*((1+g)*d*d-3*v)/v**3).sum())
    if .5*ht<=1e-7 or T<=0:
        return None
    ratio=(T-.5)/T
    den=1-.5*ratio*(1-htt/ht**2)
    step=ratio/(ht*den)
    predicted=ht+htt*step+.5*httt*step**2
    if not (np.isfinite(step) and np.isfinite(predicted) and den>0 and predicted>2e-7
            and abs(step)<.5*np.min(abs(t-p)) and abs(step)<.0002):
        return None
    return t+step,-quantities(t+step,a,p,w)[3]/predicted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('Fresh output required.')
    a,r,z=sp.symbols('a r z',positive=True)
    g=a*sp.exp(-r*r*z*z/2)
    R=g*z*r*r/(4*(1-g))
    ddR=g*z*r**4*((1+g)*r*r*z*z-3*(1-g))/(4*(1-g)**3)
    assert sp.simplify(sp.diff(R,z,2)-ddR)==0
    assert sp.simplify(sp.diff(-g*z*r*r/(4*(1-g)),z,2)+ddR)==0
    F,T,ht,htt=sp.symbols('F T ht htt',nonzero=True)
    halley=-2*F*(-T*ht)/(2*(-T*ht)**2-F*T*(ht**2-htt))
    formula=(F/T)/(ht*(1-sp.Rational(1,2)*(F/T)*(1-htt/ht**2)))
    assert sp.cancel(halley-formula)==0
    cases=[([.8,.65],[.8,1.2],[.03,.05]),([.4,.7,.3],[.4,.5,.6],[.05,.08,.04]),
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
            result=refine(root+e,aa,pp,ww)
            if result is None:
                raise ArithmeticError('Smooth oracle unexpectedly rejected refinement.')
            t1,jac=result
            rows.append({'initial_root_error':e,'refined_root_error':abs(t1-root),
                'refined_jacobian_relative_l2_error':float(np.linalg.norm(jac-oracle)/np.linalg.norm(oracle))})
        for j in range(1,len(rows)):
            rows[j]['jacobian_observed_order']=float(np.log2(rows[j-1]['refined_jacobian_relative_l2_error']/rows[j]['refined_jacobian_relative_l2_error']))
        # Last point can approach float64 precision; enforce order before that floor.
        assert rows[2]['jacobian_observed_order']>2.8
        records.append({'alpha':aa.tolist(),'peaks':pp.tolist(),'widths':ww.tolist(),'root':root,'rows':rows})
    rejected=refine(1e-4,np.array([.75]),np.array([0.]),np.array([.1])) is None
    assert rejected
    result={'symbolic_branch_second_derivative_pass':True,'symbolic_halley_identity_pass':True,
        'smooth_cases':records,'critical_example_rejected':rejected,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Separated-peak float64 effective-ray Jacobian oracle; not CUDA or a full training result.',
        'The coarse primal still gives first-order upstream loss derivative error in general.',
        'Guard acceptance is a local engineering check, not a formal root or floating-point certificate.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'smooth_cases':len(records),'critical_rejected':rejected,
        'min_midrange_order':min(c['rows'][2]['jacobian_observed_order'] for c in records)}))


if __name__=='__main__':
    main()
