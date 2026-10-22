"""Exact simplification of GGGS affine camera normal, with independent AD checks."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sympy as sp
import torch
from verify_inverse_covariance_vjp import rotation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required.')
    u,v=sp.symbols('u v',real=True);z=sp.symbols('z',positive=True)
    a,b,c,d,e,f=sp.symbols('a b c d e f',real=True)
    P=sp.Matrix([[a,b,c],[b,d,e],[c,e,f]]);r=sp.Matrix([u,v,1]);q=P*r;L=1+u*u+v*v;vb=(r.T*q)[0]
    K=sp.Matrix([[1+v*v,-u*v,-u],[-u*v,1+u*u,-v],[0,0,0]])
    J=sp.Matrix([[1/z,0,u/sp.sqrt(L)],[0,1/z,v/sp.sqrt(L)],[-u/z,-v/z,1/sp.sqrt(L)]])
    plane=K*q/vb;factor=z/sp.sqrt(L)
    raw=J*sp.Matrix([-plane[0]*factor,-plane[1]*factor,-1])
    identities=[sp.simplify(x)==0 for x in raw+sp.sqrt(L)*q/vb]
    if not all(identities):raise AssertionError('Exact affine normal identity failed.')
    torch.manual_seed(53);rows=[]
    for i in range(64):
        t=torch.tensor([.6*torch.randn(()),.6*torch.randn(()),1.5+torch.rand(())],dtype=torch.float64,requires_grad=True)
        ls=(-3*torch.rand(3,dtype=torch.float64)).requires_grad_(True)
        quat=torch.randn(4,dtype=torch.float64,requires_grad=True)
        R=rotation(quat);M=torch.diag(torch.exp(-ls))@R;precision=M.T@M
        u,v=t[0]/t[2],t[1]/t[2];zero=u*0;one=zero+1;ray=torch.stack((u,v,one))
        L=1+u*u+v*v;qt=precision@ray;vb=(ray*qt).sum()
        K=torch.stack((1+v*v,-u*v,-u,-u*v,1+u*u,-v,zero,zero,zero)).reshape(3,3)
        J=torch.stack((1/t[2],zero,u/torch.sqrt(L),zero,1/t[2],v/torch.sqrt(L),-u/t[2],-v/t[2],1/torch.sqrt(L))).reshape(3,3)
        plane=K@qt/vb;factor=t[2]/torch.sqrt(L)
        raw=J@torch.stack((-plane[0]*factor,-plane[1]*factor,-one))
        old=raw/torch.linalg.vector_norm(raw);simple=-qt/torch.linalg.vector_norm(qt)
        g=torch.randn(3,dtype=torch.float64)
        go=torch.cat(torch.autograd.grad((old*g).sum(),(t,ls,quat),retain_graph=True))
        gs=torch.cat(torch.autograd.grad((simple*g).sum(),(t,ls,quat)))
        norm=max(float(torch.linalg.vector_norm(go)),1e-12)
        rows.append({'case':i,'normal_max_abs_difference':float((old-simple).abs().max().detach()),
            'world_vjp_relative_l2':float(torch.linalg.vector_norm(gs-go))/norm,
            'raw_normal_length':float(torch.linalg.vector_norm(raw).detach()),
            'incidence_reciprocal':float((torch.linalg.vector_norm(qt)*torch.sqrt(L)/vb).detach())})
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'exact_component_identities':identities,
        'identity':'unnormalized c=-P*d/(d^T*P*d), unit d; normalized n=-P*d/||P*d||',
        'length_identity':'||c||=1/(abs(n dot d)) >= 1 for symmetric positive-definite P',
        'correct_world_differential':'dn=-(I-n*n^T)*(dP*d+P*dd)/||P*d||',
        'rows':rows,'max_world_vjp_relative_l2':max(r['world_vjp_relative_l2'] for r in rows),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['Exact algebra on the positive-definite branch, positive camera z and nonzero vb; active FoV clipping must also be differentiated.',
            'Two independent float64 computational graphs, not an audit of the CUDA world VJP.',
            'No implementation, runtime, training, quality or novelty claim from the simplification.']}
    if report['max_world_vjp_relative_l2']>1e-10:raise AssertionError('Independent world VJP graphs differ.')
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Exact affine-normal identity and 64 independent world AD checks passed.',flush=True)


if __name__=='__main__':main()
