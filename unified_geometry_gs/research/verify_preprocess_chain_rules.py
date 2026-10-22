"""Exact chain rules for GGGS ray precision and opacity filtering coefficient."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sympy as sp
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required.')
    u,v=sp.symbols('u v',real=True)
    a,b,c,e,f,g=sp.symbols('a b c e f g',real=True)
    P=sp.Matrix([[a,b,c],[b,e,f],[c,f,g]]);r=sp.Matrix([u,v,1]);h=P*r
    L=1+u*u+v*v;B=(r.T*h)[0];sigma=sp.sqrt(B/L)
    ray_identities=[sp.simplify(sp.diff(sigma,x)-(h[j]/(sigma*L)-sigma*x/L))==0 for j,x in enumerate((u,v))]
    d0,d1=sp.symbols('d0 d1',positive=True);coef=sp.sqrt(d0/d1)
    coef_identity=sp.simplify(sp.diff(coef,d1)+sp.diff(coef,d0)*coef**2)==0
    isotropic=sp.simplify(sigma.subs({a:1,e:1,g:1,b:0,c:0,f:0}))
    isotropic_legacy=sp.simplify((-sigma*u/L).subs({a:1,e:1,g:1,b:0,c:0,f:0,u:sp.Rational(1,2),v:0}))
    if ray_identities!=[True,True] or not coef_identity or isotropic!=1 or isotropic_legacy!=-sp.Rational(2,5):
        raise AssertionError('Exact chain-rule identity or counterexample failed.')
    torch.manual_seed(83);rows=[]
    for i in range(64):
        M=torch.randn(3,3,dtype=torch.float64);precision=M.T@M+torch.eye(3,dtype=torch.float64)
        uv=torch.randn(2,dtype=torch.float64,requires_grad=True)
        ray=torch.cat((uv,torch.ones(1,dtype=torch.float64)));length=(ray*ray).sum();h=precision@ray
        s=torch.sqrt((h*ray).sum()/length);automatic=torch.autograd.grad(s,uv)[0]
        analytic=h[:2]/(s.detach()*length.detach())-s.detach()*uv.detach()/length.detach()
        x=(torch.rand(2,dtype=torch.float64)+.1).requires_grad_(True)
        cc=torch.sqrt(x[0]/x[1]);auto=torch.autograd.grad(cc,x)[0]
        ddc0=1/(2*cc.detach()*x[1].detach());correct=torch.stack((ddc0,-ddc0*cc.detach()**2))
        rows.append({'case':i,'ray_derivative_max_abs':float((automatic-analytic).detach().abs().max()),
            'coef_derivative_max_abs':float((auto-correct).detach().abs().max())})
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'exact_ray_identities':ray_identities,
        'exact_coef_identity':coef_identity,'ray_formula':'d sigma / du_i = (P*r)_i/(sigma*L) - sigma*u_i/L, L=r^T*r',
        'coefficient_formula':'d c / d D1 = -(d c / d D0)*c^2, c=sqrt(D0/D1)',
        'counterexample':{'P':'identity','u':.5,'v':0.,'sigma':1.,'true_dsigma_du':0.,'legacy_dsigma_du':-.4},
        'rows':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'domains':['P symmetric positive definite, positive B and sigma; no active backward vb clamp.',
                   'Positive unclamped D0,D1. Existing det floors and coefficient epsilon are distinct stabilization choices.'],
        'limitations':['Local exact identities and independent float64 AD checks; no reconstruction quality or novelty claim.']}
    if max(max(r['ray_derivative_max_abs'],r['coef_derivative_max_abs']) for r in rows)>1e-10:raise AssertionError('Float64 check failed.')
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Three exact chain-rule identities, isotropic counterexample and 64 float64 cases passed.',flush=True)


if __name__=='__main__':main()
