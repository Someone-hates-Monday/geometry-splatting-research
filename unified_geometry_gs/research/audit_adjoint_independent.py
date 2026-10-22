"""CPU-only independent product/autograd oracle on stored real-ray parameters.

Uses the previously selected failing development camera. This is an effective
ray audit, not a full world-parameter CUDA derivative correctness proof.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import torch
from research.verify_adjoint_refinement import quantities
from research.verify_adjquad_refinement import refine


def product_T(t,a,p,w):
    g=a*np.exp(-.5*((t-p)/w)**2)
    v=np.sqrt(1-g)
    return float(np.prod(np.where(t<=p,v,(1-a)/v)))


def root_product(a,p,w):
    lo=float(np.min(p)-16*np.max(w))
    hi=float(np.max(p)+16*np.max(w))
    if not (product_T(lo,a,p,w)>.5 and product_T(hi,a,p,w)<.5):
        raise ArithmeticError('Direct product bracket invalid; do not silently reuse log-field oracle.')
    return brentq(lambda t:product_T(t,a,p,w)-.5,lo,hi,xtol=1e-14)


def autograd_jac(t,a,p,w):
    t=torch.tensor(t,dtype=torch.float64,requires_grad=True)
    a,p,w=(torch.tensor(x,dtype=torch.float64,requires_grad=True) for x in (a,p,w))
    g=a*torch.exp(-.5*((t-p)/w)**2)
    v=torch.sqrt(1-g)
    T=torch.where(t<=p,v,(1-a)/v).prod()
    dt,da,dp,dw=torch.autograd.grad(T,(t,a,p,w))
    return np.r_[(-da/dt).detach().numpy(),(-dp/dt).detach().numpy(),(-dw/dt).detach().numpy()],float(T.detach()),float(dt.detach())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output required.')
    torch.set_num_threads(1)
    original=json.loads(args.input.read_text())
    rng=np.random.default_rng(0)
    rows=[]
    finite_diff=[]
    for index,row in enumerate(original['rows']):
        a,p,w=(np.array(row[k],dtype=float) for k in ('alpha','peaks','widths'))
        root=root_product(a,p,w)
        true,T,Tt=autograd_jac(root,a,p,w)
        _,ht,_,num=quantities(root,a,p,w)
        analytic=-num/ht
        norm=max(float(np.linalg.norm(true)),1e-15)
        def relative(j):return float(np.linalg.norm(j-true)/norm)
        def approx(t):
            _,h,_,n=quantities(t,a,p,w)
            return -n/h
        t0=row['coarse_ray_depth']
        refined=refine(t0,a,p,w)
        record={'pixel':row['pixel'],'contributors':len(a),'direct_product_root':root,
            'root_difference_from_stored_cpu_reference':abs(root-row['reference_ray_depth']),
            'root_product_residual':abs(T-.5),'T_t':Tt,
            'root_nearest_peak_distance':float(np.min(abs(root-p))),
            'analytic_vs_product_autograd_relative_l2':relative(analytic),
            'coarse_effective_jacobian_relative_l2':relative(approx(t0)),
            'five_effective_jacobian_relative_l2':relative(approx(row['control_ray_depth'])),
            'halley_guard_accept':refined is not None,
            'idealized_halley_effective_jacobian_relative_l2':None if refined is None else relative(refined[1]),
            'halley_refined_root_error':None if refined is None else abs(refined[0]-root)}
        rows.append(record)
        # The first twelve rays are the pre-existing largest coarse-error rays.
        # Directions are scaled logit-alpha, sigma-scaled peak and log-sigma.
        if index<12:
            n=len(a)
            direction=rng.normal(size=3*n)
            direction/=np.linalg.norm(direction)
            da,dp,dw=np.split(direction,3)
            raw_dir=np.r_[a*(1-a)*da,w*dp,w*dw]
            oracle=float(true@raw_dir)
            logit=np.log(a/(1-a))
            records=[]
            for h in (1e-4,1e-5,1e-6):
                roots=[]
                for sign in (-1,1):
                    aa=1/(1+np.exp(-(logit+sign*h*da)))
                    pp=p+sign*h*w*dp
                    ww=w*np.exp(sign*h*dw)
                    roots.append(root_product(aa,pp,ww))
                fd=(roots[1]-roots[0])/(2*h)
                records.append({'h':h,'finite_difference':fd,'absolute_error':abs(fd-oracle)})
            middle=records[1]
            threshold=1e-7+1e-4*abs(oracle)
            finite_diff.append({'pixel':row['pixel'],'directional_autograd':oracle,'rows':records,
                'fixed_h_1e_5_tolerance':threshold,'fixed_h_1e_5_pass':middle['absolute_error']<=threshold})
    checks={'all_analytic_vs_independent_product_autograd_relative_l2_le_1e_10':all(r['analytic_vs_product_autograd_relative_l2']<=1e-10 for r in rows),
        'all_fixed_middle_step_finite_differences_pass':all(r['fixed_h_1e_5_pass'] for r in finite_diff),
        'torch_cuda_not_initialized':not torch.cuda.is_initialized()}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'model':original['model'],
        'camera':original['camera'],'rays':len(rows),'rows':rows,'finite_differences':finite_diff,'checks':checks,
        'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.input,Path(__file__),Path(__file__).with_name('verify_adjoint_refinement.py'),Path(__file__).with_name('verify_adjquad_refinement.py'))},
        'limitations':['One previously used development camera and its fixed sampled effective-ray parameters.',
        'Product autograd and re-solved finite differences are independent of analytic H derivatives, but share stored affine ray parameters.',
        'Idealized Halley refinement is float64; CUDA float32 contributor selection, parameters, guards and world-gradient propagation remain separate.',
        'Five/coarse roots came from CUDA with float32 affine parameters; their error vs this float64 field includes projection/roundoff mismatch.',
        'These are correlated pixels, not independent training/scene replications or reconstruction quality evidence.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    selected=[r for r in rows if r['halley_guard_accept']]
    print(json.dumps({'rays':len(rows),'accepted_halley':len(selected),'checks':checks,
        'max_analytic_vs_autograd':max(r['analytic_vs_product_autograd_relative_l2'] for r in rows),
        'max_five_vs_autograd':max(r['five_effective_jacobian_relative_l2'] for r in rows),
        'max_idealized_halley_vs_autograd':max(r['idealized_halley_effective_jacobian_relative_l2'] for r in selected),
        'max_fixed_middle_step_absolute_error':max(r['rows'][1]['absolute_error'] for r in finite_diff)},indent=2))
    if not all(checks.values()):raise ArithmeticError('Independent oracle audit failed; preserve report and investigate.')


if __name__=='__main__':main()
