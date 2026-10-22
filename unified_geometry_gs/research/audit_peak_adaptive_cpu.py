"""Development feasibility: restore pass five when the final cell contains a peak.

CPU float64 field-point counts are algorithmic proxies, never GPU timings.
All 123 input rays are retained, including rejected adjoint corrections.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from research.audit_adjoint_independent import root_product,autograd_jac
from research.verify_adjoint_refinement import quantities
from research.verify_adjquad_refinement import refine


def T_many(t,a,p,w):
    t=np.asarray(t)[:,None]
    g=a*np.exp(-.5*((t-p)/w)**2)
    v=np.sqrt(1-g)
    return np.prod(np.where(t<=p,v,(1-a)/v),axis=1)


def subdivision(a,p,w,passes):
    before=np.r_[1.,np.cumprod(1-a)[:-1]]
    init=p[np.flatnonzero(before>.5)[-1]]
    lo,hi=max(init-.4,0.),max(init+.4,0.)
    values=None
    count=0
    history=[]
    for j in range(passes):
        width=(hi-lo)/8
        points=lo+np.arange(9)*width
        if j==0:
            values=T_many(points,a,p,w);count+=9
            if not (values[0]>=.5 and values[-1]<=.5):raise ArithmeticError('Initial bracket invalid.')
        else:
            values=np.r_[values[0],T_many(points[1:-1],a,p,w),values[-1]];count+=7
        start=0
        for k in range(1,8):
            if values[k]>=.5:start=k
        lo,hi=points[start],points[start+1]
        values=np.r_[values[start],values[start+1]]
        weight=np.clip((values[0]-.5)/(values[0]-values[1]),0,1)
        history.append({'lo':float(lo),'hi':float(hi),'depth':float((1-weight)*lo+weight*hi),
            'T_lo':float(values[0]),'T_hi':float(values[1]),'field_evaluations':count})
    return history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output required.')
    data=json.loads(args.input.read_text())
    rows=[]
    for row in data['rows']:
        a,p,w=(np.array(row[k]) for k in ('alpha','peaks','widths'))
        history=subdivision(a,p,w,5)
        four,five=history[3:5]
        flagged=bool(((p>=four['lo'])&(p<=four['hi'])).any())
        t_adaptive=five['depth'] if flagged else four['depth']
        root=root_product(a,p,w)
        true,_,_=autograd_jac(root,a,p,w)
        norm=max(float(np.linalg.norm(true)),1e-15)
        def effective(t):
            _,ht,_,num=quantities(t,a,p,w)
            return -num/ht
        def adjoint(t):
            r=refine(t,a,p,w)
            return (effective(t),False) if r is None else (r[1],True)
        plain,plain_accept=adjoint(four['depth'])
        adaptive,adaptive_accept=adjoint(t_adaptive)
        rel=lambda j:float(np.linalg.norm(j-true)/norm)
        rows.append({'pixel':row['pixel'],'four_cell':four,'five_cell':five,
            'contains_peak':flagged,'plain_guard_accept':plain_accept,'adaptive_guard_accept':adaptive_accept,
            'plain_adjoint_with_fallback_relative_l2':rel(plain),
            'adaptive_adjoint_with_fallback_relative_l2':rel(adaptive),
            'five_implicit_relative_l2':rel(effective(five['depth'])),
            'adaptive_field_evaluations':37 if flagged else 30})
    def stats(key):
        v=np.array([r[key] for r in rows])
        return {'p50':float(np.median(v)),'p95':float(np.quantile(v,.95)),'p99':float(np.quantile(v,.99)),'max':float(v.max())}
    summary={'rays':len(rows),'peak_flagged_rays':sum(r['contains_peak'] for r in rows),
        'plain_guard_rejections':sum(not r['plain_guard_accept'] for r in rows),
        'adaptive_guard_rejections':sum(not r['adaptive_guard_accept'] for r in rows),
        'plain_with_fallback_jacobian_errors':stats('plain_adjoint_with_fallback_relative_l2'),
        'adaptive_with_fallback_jacobian_errors':stats('adaptive_adjoint_with_fallback_relative_l2'),
        'five_jacobian_errors':stats('five_implicit_relative_l2'),
        'mean_adaptive_field_evaluations':float(np.mean([r['adaptive_field_evaluations'] for r in rows])),
        'fixed_four_field_evaluations':30,'fixed_five_field_evaluations':37}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'model':data['model'],'camera':data['camera'],
        'policy':'After four passes, restore pass five iff at least one contributing peak lies in the final closed cell; retain original high-order adjoint guard.',
        'summary':summary,'rows':rows,
        'source_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.input,Path(__file__),Path(__file__).with_name('verify_adjquad_refinement.py'),Path(__file__).with_name('audit_adjoint_independent.py'))},
        'limitations':['One previously used development camera with 123 stored effective-ray parameter lists.',
            'Float64 CPU subdivision and adjoint, not CUDA masks/guard parity or full world gradients.',
            'Field-point counts omit tile barriers, memory movement, extra peak detection and divergent-thread costs.',
            'No GPU acceleration, quality, or novelty claim; this policy is a development hypothesis.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
