"""Check Halley's local coefficient and a GGGS loss-gradient order limitation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import sympy as sp
from research.verify_adjquad_refinement import refine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output required.')
    u,a,b=sp.symbols('u a b')
    f=u+a*u**2+b*u**3
    error=u-2*f*sp.diff(f,u)/(2*sp.diff(f,u)**2-f*sp.diff(f,u,2))
    coefficient=sp.expand(sp.series(error,u,0,4).removeO()).coeff(u,3)
    assert sp.cancel(coefficient-(a*a-b))==0
    # Single regular GGGS primitive. Moving p translates t* exactly, so J_p=1.
    # A quadratic depth loss gives upstream factor t0; that coarse factor alone
    # prevents the complete gradient from having cubic accuracy.
    aa,pp,ww=np.array([.9]),np.array([1.]),np.array([.1])
    root=1-.1*np.sqrt(-2*np.log(.75/.9))
    rows=[]
    for e in (1e-4,5e-5,2.5e-5,1.25e-5,6.25e-6):
        out=refine(root+e,aa,pp,ww)
        if out is None:raise ArithmeticError('Regular example rejected.')
        t1,jac=out
        jp=float(jac[1])
        rows.append({'coarse_root_error':e,'refined_root_error':abs(t1-root),
            'peak_jacobian_error':abs(jp-1),
            'coarse_upstream_loss_gradient_error':abs((root+e)*jp-root)})
    for i in range(1,len(rows)):
        rows[i]['loss_gradient_observed_order']=float(np.log2(rows[i-1]['coarse_upstream_loss_gradient_error']/rows[i]['coarse_upstream_loss_gradient_error']))
    assert all(.95<r['loss_gradient_observed_order']<1.05 for r in rows[1:])
    result={'symbolic_halley_cubic_coefficient_pass':True,'coefficient':str(coefficient),
        'GGGS_single_primitive':{'alpha':.9,'peak':1.,'sigma':.1,'q':.5,'exact_root':float(root),'exact_peak_jacobian':1.,
            'loss':'0.5 * depth^2','exact_loss_gradient_peak':float(root)},'rows':rows,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('verify_adjquad_refinement.py'))},
        'interpretation':'A regular GGGS example: a high-order local Jacobian does not give high-order complete loss gradient when upstream remains coarse.',
        'limitations':['Float64 single-primitive mechanism check, not an assertion about the size or direction of actual NCC training bias.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'symbolic_pass':True,'loss_gradient_order_range':[
        min(r['loss_gradient_observed_order'] for r in rows[1:]),max(r['loss_gradient_observed_order'] for r in rows[1:])]}))


if __name__=='__main__':main()
