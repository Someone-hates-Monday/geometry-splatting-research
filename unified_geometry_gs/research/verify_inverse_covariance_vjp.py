"""Exact scale identity and independent autograd comparison for inverse-factor VJP."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import torch
import sympy as sp


def rotation(q):
    r,x,y,z=q/torch.linalg.vector_norm(q)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-r*z),2*(x*z+r*y),
        2*(x*y+r*z),1-2*(x*x+z*z),2*(y*z-r*x),
        2*(x*z-r*y),2*(y*z+r*x),1-2*(x*x+y*y))).reshape(3,3)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Fresh output required.')
    a,b,c=sp.symbols('a b c',positive=True)
    # Rational orthogonal rotation from the exact normalized quaternion (1,2,3,4).
    R=sp.Matrix([[-sp.Rational(2,3),sp.Rational(2,15),sp.Rational(11,15)],
                 [sp.Rational(2,3),-sp.Rational(1,3),sp.Rational(2,3)],
                 [sp.Rational(1,3),sp.Rational(14,15),sp.Rational(2,15)]])
    if R.T*R!=sp.eye(3):raise AssertionError('Exact rotation is not orthogonal.')
    S=sp.diag(a,b,c);cov=R.T*S*S*R;inv=R.T*S.inv()*S.inv()*R
    G=sp.Matrix(3,3,sp.symbols('g0:9'))
    loss=sp.trace(G.T*inv)
    Gcov=-inv*G*inv
    identity=[]
    for s in (a,b,c):
        direct=sp.diff(loss,s)
        chain=sp.trace(Gcov.T*cov.diff(s))
        identity.append(sp.simplify(direct-chain)==0)
    if not all(identity):raise AssertionError('Exact scale identity failed.')
    torch.manual_seed(17);rows=[]
    for aspect in (1.,1e2,1e4,1e6):
        for seed in range(12):
            q0=torch.randn(4,dtype=torch.float64)
            g0=torch.randn(3,3,dtype=torch.float64)
            s0=torch.tensor([.1,.07,.1/aspect],dtype=torch.float64)
            results={}
            for dtype in (torch.float64,torch.float32):
                ls=s0.log().to(dtype).requires_grad_(True);q=q0.to(dtype).requires_grad_(True)
                r=rotation(q);s=ls.exp();m=torch.diag(s)@r;mi=torch.diag(1/s)@r
                sigma=m.T@m;precision=mi.T@mi;g=g0.to(dtype)
                oracle=torch.autograd.grad((precision*g).sum(),(ls,q),retain_graph=True)
                gcov=(-precision@g@precision).detach()
                chain=torch.autograd.grad((sigma*gcov).sum(),(ls,q),retain_graph=True)
                gm=mi@(g+g.T)
                # Direct VJP in the inverse factor; differentiates only its inputs.
                scale_direct=-torch.diagonal(gm@r.T)/s
                grad_r=torch.diag(1/s)@gm
                rot_direct=torch.autograd.grad((r*grad_r.detach()).sum(),q)[0]
                results[str(dtype)]={'oracle':torch.cat(oracle).double(),'chain':torch.cat(chain).double(),
                                    'direct':torch.cat((scale_direct,rot_direct)).detach().double()}
            target=results['torch.float64']['oracle'];norm=max(float(torch.linalg.vector_norm(target)),1e-30)
            row={'aspect_ratio':aspect,'seed_index':seed,
                'float64_chain_relative_l2':float(torch.linalg.vector_norm(results['torch.float64']['chain']-target))/norm,
                'float64_direct_relative_l2':float(torch.linalg.vector_norm(results['torch.float64']['direct']-target))/norm,
                'float32_chain_relative_l2':float(torch.linalg.vector_norm(results['torch.float32']['chain']-target))/norm,
                'float32_direct_relative_l2':float(torch.linalg.vector_norm(results['torch.float32']['direct']-target))/norm}
            rows.append(row)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'sympy_version':sp.__version__,
        'exact_scale_chain_identity':identity,'rows':rows,
        'max_float64_direct_relative_l2':max(r['float64_direct_relative_l2'] for r in rows),
        'limitations':['Synthetic inverse-covariance linear losses only, not real renderer parameter derivatives.',
            'The inverse identity uses a positive scale and orthogonal normalized-quaternion rotation.',
            'Direct factor differentiation is standard matrix calculus; no novelty claim.',
            'Does not yet prove the observed complete-renderer variability is caused by this computation.'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if report['max_float64_direct_relative_l2']>1e-12:raise AssertionError('Direct VJP oracle mismatch.')
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Exact scale identities and direct float64 VJP passed.',flush=True)


if __name__=='__main__':main()
