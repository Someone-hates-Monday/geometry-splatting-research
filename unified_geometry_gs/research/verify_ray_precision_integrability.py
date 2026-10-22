"""Prove and check artificial angular bias and non-integrability of legacy ray VJP.

The upstream of kappa is fixed to one and all other preprocessing upstreams
are zero. This is not a statement that complete scene-center gradients vanish.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import sympy as sp
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required.')
    u,v=sp.symbols('u v',real=True);a,b,c,e,f,g=sp.symbols('a b c e f g',real=True)
    P=sp.Matrix([[a,b,c],[b,e,f],[c,f,g]]);r=sp.Matrix([u,v,1]);L=(r.T*r)[0];h=P*r
    k=sp.sqrt((r.T*h)[0]/L);legacy=sp.Matrix([-k*u/L,-k*v/L])
    curl=sp.diff(legacy[1],u)-sp.diff(legacy[0],v)
    formula=(u*h[1]-v*h[0])/(k*L**2)
    curl_identity=sp.simplify(curl-formula)==0
    sub={a:1,e:2,g:3,b:0,c:0,f:0,u:sp.Rational(1,2),v:sp.Rational(1,2)}
    curl_at=sp.simplify(formula.subs(sub))
    if not curl_identity or sp.simplify(curl_at-sp.sqrt(10)/45)!=0:raise AssertionError('Exact curl identity failed.')
    x,y=sp.symbols('x y',real=True);z,K=sp.symbols('z K',positive=True)
    radius2=x*x+y*y+z*z;potential=K*(sp.log(z)-sp.log(radius2)/2)
    radial_legacy=sp.Matrix([-K*x/radius2,-K*y/radius2,K/z-K*z/radius2])
    angular_identity=[sp.simplify(sp.diff(potential,t)-radial_legacy[j])==0 for j,t in enumerate((x,y,z))]
    norm_identity=sp.simplify((radial_legacy.T*radial_legacy)[0]-K*K*(x*x+y*y)/(z*z*radius2))==0
    if not all(angular_identity) or not norm_identity:raise AssertionError('Angular-bias identity failed.')
    torch.manual_seed(97);ad=[]
    for i in range(64):
        M=torch.randn(3,3,dtype=torch.float64);precision=M.T@M+torch.eye(3,dtype=torch.float64)
        uv=torch.randn(2,dtype=torch.float64,requires_grad=True);ray=torch.cat((uv,torch.ones(1,dtype=torch.float64)))
        ell=(ray*ray).sum();hr=precision@ray;kap=torch.sqrt((ray*hr).sum()/ell)
        wrong=-kap*uv/ell
        jac=torch.stack([torch.autograd.grad(wrong[j],uv,retain_graph=True)[0] for j in range(2)])
        value=jac[1,0]-jac[0,1];expected=(uv[0]*hr[1]-uv[1]*hr[0])/(kap*ell**2)
        ad.append({'case':i,'curl':float(value.detach()),'formula':float(expected.detach()),
                   'absolute_error':float(abs(value-expected).detach())})
    nodes,weights=np.polynomial.legendre.leggauss(64);nodes=(nodes+1)/2;weights=weights/2
    precision=np.diag([1.,2.,3.]);loops=[]
    def field(points,correct):
        rays=np.column_stack((points,np.ones(len(points))));ell=np.sum(rays*rays,axis=1)
        hr=rays@precision;kap=np.sqrt(np.sum(rays*hr,axis=1)/ell)
        result=-kap[:,None]*points/ell[:,None]
        if correct:result+=hr[:,:2]/(kap*ell)[:,None]
        return result
    for side in (.1,.03,.01,.003,.001):
        corners=np.array([[.5-side/2,.5-side/2],[.5+side/2,.5-side/2],
                          [.5+side/2,.5+side/2],[.5-side/2,.5+side/2]])
        values={}
        for correct in (False,True):
            integral=0.
            for i in range(4):
                origin=corners[i];delta=corners[(i+1)%4]-origin
                points=origin[None,:]+nodes[:,None]*delta[None,:]
                integral+=float(np.sum(weights*np.sum(field(points,correct)*delta,axis=1)))
            values['corrected' if correct else 'legacy']=integral
        loops.append({'counterclockwise_square_side':side,'area':side*side,'circulation':values,
                      'legacy_circulation_per_area':values['legacy']/(side*side)})
    expected_curl=float(curl_at)
    if max(r['absolute_error'] for r in ad)>1e-10 or abs(loops[-1]['legacy_circulation_per_area']-expected_curl)>1e-6 or max(abs(r['circulation']['corrected']) for r in loops)>1e-12:
        raise AssertionError('Independent numerical integrability check failed.')
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'exact_curl_identity':curl_identity,'curl_formula':'d_u g_v - d_v g_u = (u*(P*r)_y-v*(P*r)_x)/(kappa*(r^T*r)^2)',
        'spd_counterexample':{'P_diagonal':[1,2,3],'u':.5,'v':.5,'curl_exact':str(curl_at),'curl_float':expected_curl},
        'isotropic_angular_identity':angular_identity,'isotropic_bias_norm_identity':norm_identity,
        'isotropic_potential':'kappa*log(cos(theta)), although true kappa is constant in center direction',
        'isotropic_bias_norm':'kappa*tan(theta)/||t|| for positive camera z',
        'autograd_cases':ad,'closed_loop_quadrature':loops,
        'mathematical_implication':'For fixed anisotropic P and fixed unit kappa upstream, the legacy viewport VJP has nonzero curl and cannot be the gradient of any C2 local scalar potential in a neighborhood of the stated point.',
        'limits':['Interior FoV, symmetric positive-definite P, positive camera z, fixed other output upstreams at zero.',
            'This is about the ray-precision component, not the complete world-center gradient or image/scene rotation invariance.',
            'An artificial isotropic angular term is a derivative mechanism, not an experimentally established cause of reconstruction errors.',
            'Clairaut symmetry and circulation tests are standard calculus. No novelty or training-quality claim.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'exact_curl':str(curl_at),'curl_numeric':expected_curl,'last_legacy_circulation_per_area':loops[-1]['legacy_circulation_per_area'],
        'max_corrected_abs_circulation':max(abs(r['circulation']['corrected']) for r in loops)},indent=2),flush=True)


if __name__=='__main__':main()
