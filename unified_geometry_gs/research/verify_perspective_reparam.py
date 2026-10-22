"""Symbolic and fixed synthetic checks of an affine-to-perspective reparameterization."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import sympy as sp
from research.perspective_reparam import recover

BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def symbolic():
    a,c,e,f,h,b=sp.symbols('a c e f h b',real=True,nonzero=True)
    Q=sp.Matrix([[a,c,e],[c,f,h],[e,h,b]])
    projected_cov=Q.inv()[:2,:2]
    schur=Q[:2,:2]-Q[:2,2:3]*Q[2:3,:2]/b
    checks={'schur_projection_identity':all(sp.factor(x)==0 for x in projected_cov.inv()-schur)}
    u,v,dx,dy,z=sp.symbols('u v dx dy z',real=True)
    P=Q;q0=sp.Matrix([u,v,1]);delta=sp.Matrix([dx,dy,0]);q=q0+delta
    M=sp.Matrix([[1,0,u],[0,1,v],[0,0,1]])
    J=sp.Matrix([[1,0,-u],[0,1,-v]])
    checks['general_center_projection_basis']=J*M==sp.Matrix([[1,0,0],[0,1,0]]) and M.det()==1
    B=(q0.T*P*q0)[0];cross=(delta.T*P*q0)[0];H=(delta.T*P*delta)[0]
    r=(q0.T*q0)[0];g=(q0.T*delta)[0]/r;k=(delta.T*delta)[0]/r
    ell=cross/B;chi=H/B-ell**2;D=(1+ell)**2+chi
    exactden=(q.T*P*q)[0]
    checks['ray_length_ratio']=sp.factor(1+2*g+k-(q.T*q)[0]/r)==0
    checks['peak_ratio']=sp.factor((1+ell)/D-(q.T*P*q0)[0]/exactden)==0
    checks['width_squared']=sp.factor((r/B)*(1+2*g+k)/D-(q.T*q)[0]/exactden)==0
    energy=z*z*(H-cross*cross/B)
    checks['peak_energy']=sp.factor(energy/D-z*z*(B-(q.T*P*q0)[0]**2/exactden))==0
    tau=sp.symbols('tau',real=True);qt=q0+tau*delta
    peak=z*sp.sqrt((qt.T*qt)[0])*(qt.T*P*q0)[0]/(qt.T*P*qt)[0]
    shift=z*((q0.T*delta)[0]/sp.sqrt(r)-sp.sqrt(r)*cross/B)
    checks['affine_is_peak_Taylor_term']=sp.factor(sp.diff(peak,tau).subs(tau,0)-shift)==0
    checks['ell_from_linear_peak_shift']=sp.factor(g-shift/(z*sp.sqrt(r))-ell)==0
    checks['chi_from_screen_energy']=sp.factor(energy*(r/B)/(z*z*r)-chi)==0
    return checks

def synthetic():
    rng=np.random.default_rng(13015);rows=[]
    for i in range(64):
        rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
        axes=10**rng.uniform(-2,0,3);variance=axes**2
        P=(rotation/variance)@rotation.T;cov=(rotation*variance)@rotation.T
        q0=np.r_[rng.uniform(-.8,.8,2),1.];delta=np.r_[rng.uniform(-.15,.15,2),0.]
        z=rng.uniform(.5,5);mean=z*q0;direction=q0+delta;direction/=np.linalg.norm(direction)
        r=q0@q0;B=q0@P@q0;t0=z*np.sqrt(r);s0=np.sqrt(r/B)
        shift=z*(q0@delta/np.sqrt(r)-np.sqrt(r)*(delta@P@q0)/B)
        J=np.array([[1.,0.,-q0[0]],[0.,1.,-q0[1]]])/z
        energy=delta[:2]@np.linalg.inv(J@cov@J.T)@delta[:2]
        recovered=recover(t0,s0,shift,energy,q0,delta,.75)
        local_o=-rotation.T@mean;local_d=rotation.T@direction
        A=np.sum(local_d**2/variance);peak=-np.sum(local_o*local_d/variance)/A
        perpendicular=local_o+peak*local_d
        expected=np.array([peak,1/np.sqrt(A),.75*np.exp(-.5*np.sum(perpendicular**2/variance))])
        errors=abs(np.array(recovered[:3])-expected)/np.maximum(1e-12,abs(expected))
        rows.append({'case':i,'axes':axes.tolist(),'rotation':rotation.tolist(),'q0':q0.tolist(),'delta':delta.tolist(),
          'z':float(z),'t0':float(t0),'s0':float(s0),'linear_peak_shift':float(shift),'energy':float(energy),
          'expected':expected.tolist(),'recovered':list(map(float,recovered[:3])),
          'scaled_errors':errors.tolist(),'pass':bool(max(errors)<=1e-8)})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Fresh output required')
    checks=symbolic();rows=synthetic()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'symbolic_checks':checks,'synthetic_cases':rows,
      'seed':13015,'numerical_gate_pass':all(checks.values()) and all(r['pass'] for r in rows),
      'max_scaled_errors':np.max([r['scaled_errors'] for r in rows],axis=0).tolist(),
      'source_sha256':{p.name:sha(p) for p in [Path(__file__),BASE/'perspective_reparam.py',BASE/'PERSPECTIVE_REPARAM_PROTOCOL.md']},
      'limits':['Exact-real identities require unclipped center projection, zero screen kernel and inverse covariance/precision.',
                'No CUDA, backward, quality, timing or novelty conclusion.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('symbolic_checks','numerical_gate_pass','max_scaled_errors')},indent=2),flush=True)
    if not result['numerical_gate_pass']:raise ArithmeticError('Fixed theoretical/synthetic gate failed; results preserved')

if __name__=='__main__':main()
