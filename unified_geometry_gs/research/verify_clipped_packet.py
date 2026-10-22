"""General symbolic checks and 128 fixed SPD tests for clipped packet recovery."""
from __future__ import annotations
import argparse,hashlib,json,platform
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from research.clipped_packet import recover

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def symbolic():
    import sympy as sp
    kx,ky,kz,cx,cy,ax,ay,mx,my,b,z=sp.symbols('kx ky kz cx cy ax ay mx my b z',real=True,nonzero=True)
    K=sp.Matrix([[kx,ky],[ky,kz]]);c=sp.Matrix([cx,cy]);a=sp.Matrix([ax,ay]);m=sp.Matrix([mx,my])
    Q=(b*(K+c*c.T)).row_join(b*c).col_join((b*c.T).row_join(sp.Matrix([[b]])))
    # Q is the original P Gram matrix in [e1,e2,qbar] coordinates.
    A=1+(c.T*a)[0];B=1+(c.T*m)[0]
    den=A*A+(a.T*K*a)[0];cross=A*B+(a.T*K*m)[0];mean=B*B+(m.T*K*m)[0]
    va=a.col_join(sp.Matrix([1]));vm=m.col_join(sp.Matrix([1]))
    v=A*m-B*a
    gdet=K.det()*(ax*my-ay*mx)**2+(v.T*K*v)[0]
    checks={'metric_quad':sp.factor((va.T*Q*va)[0]/b-den)==0,
            'metric_cross':sp.factor((va.T*Q*vm)[0]/b-cross)==0,
            'metric_mean':sp.factor((vm.T*Q*vm)[0]/b-mean)==0,
            'gram_positive_decomposition':sp.factor(den*mean-cross*cross-gdet)==0}
    projected=Q.inv()[:2,:2]/z**2
    checks['projected_conic_recovers_K']=all(sp.factor(x)==0 for x in projected.inv()-z*z*b*K)
    u,vv=sp.symbols('u vv',real=True);r=1+u*u+vv*vv
    encoded=z/sp.sqrt(r)*(r*c-sp.Matrix([u,vv]))
    checks['plane_encoding_inverse']=all(sp.factor(x)==0 for x in (encoded*sp.sqrt(r)/z+sp.Matrix([u,vv]))/r-c)
    return checks

def numeric():
    rng=np.random.default_rng(13016);rows=[]
    for i in range(128):
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        axes=10**rng.uniform(-2,0,3);variance=axes**2
        qmean=np.r_[rng.uniform(-1.8,1.8,2),1.];qbar=np.r_[np.clip(qmean[:2],-.65,.65),1.]
        qray=np.r_[rng.uniform(-.8,.8,2),1.];z=rng.uniform(.5,5);focal=np.array([800.,850.])
        P=(rotation/variance)@rotation.T;cov=(rotation*variance)@rotation.T
        rbar=qbar@qbar;pb=P@qbar;b=qbar@pb
        J=np.array([[focal[0]/z,0.,-focal[0]*qbar[0]/z],[0.,focal[1]/z,-focal[1]*qbar[1]/z]])
        conicmatrix=np.linalg.inv(J@cov@J.T);conic=np.array([conicmatrix[0,0],conicmatrix[0,1],conicmatrix[1,1]])
        linear=z/np.sqrt(rbar)*(rbar*pb/b-qbar)[:2]/focal
        plane=np.r_[linear,z*np.linalg.norm(qmean)];width=np.sqrt(rbar/b)
        peak,sigma,alpha,diag=recover(plane,np.asarray(width),conic,qmean,qbar,qray,focal,np.asarray(.75))
        direction=qray/np.linalg.norm(qray);mean=z*qmean
        local_o=-rotation.T@mean;local_d=rotation.T@direction
        quadratic=np.sum(local_d**2/variance);targetp=-np.sum(local_o*local_d/variance)/quadratic
        perpendicular=local_o+targetp*local_d
        expected=np.array([targetp,1/np.sqrt(quadratic),.75*np.exp(-.5*np.sum(perpendicular**2/variance))])
        candidate=np.array([peak,sigma,alpha]);errors=abs(candidate-expected)/np.maximum(1e-12,abs(expected))
        valid=bool(diag['detK']>0 and diag['denominator']>0 and diag['energy']>=0 and np.all(np.isfinite(candidate)))
        rows.append({'case':i,'clipped':bool(np.any(qbar!=qmean)),'qmean':qmean.tolist(),'qbar':qbar.tolist(),'qray':qray.tolist(),
          'axes':axes.tolist(),'rotation':rotation.tolist(),'z':float(z),'plane':plane.tolist(),'width':float(width),'conic':conic.tolist(),
          'expected':expected.tolist(),'candidate':candidate.tolist(),'scaled_errors':errors.tolist(),
          'valid':valid,'pass':valid and bool(max(errors)<=1e-8)})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--skip-symbolic',action='store_true');args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    checks=None if args.skip_symbolic else symbolic();rows=numeric()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'python':platform.python_version(),
      'numpy':np.__version__,'seed':13016,'symbolic_checks':checks,'symbolic_status':'not_run' if checks is None else 'executed',
      'synthetic_cases':rows,'clipped_cases':sum(r['clipped'] for r in rows),'numerical_gate_pass':all(r['pass'] for r in rows),
      'overall_gate_pass':checks is not None and all(checks.values()) and all(r['pass'] for r in rows),
      'max_scaled_errors':np.max([r['scaled_errors'] for r in rows],axis=0).tolist(),
      'source_sha256':{p.name:sha(p) for p in [Path(__file__),BASE/'clipped_packet.py',BASE/'CLIPPED_PACKET_PROTOCOL.md']},
      'limits':['Skipping symbolic checks leaves the overall gate incomplete.',
        'No production CUDA, VJP, GT, image/geometry quality, timing or novelty conclusion.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('symbolic_status','clipped_cases','numerical_gate_pass','overall_gate_pass','max_scaled_errors')},indent=2))
    if not result['numerical_gate_pass']:raise ArithmeticError('Fixed numerical gate failed; preserved')
    if checks is not None and not all(checks.values()):raise ArithmeticError('Symbolic checks failed; preserved')

if __name__=='__main__':main()
