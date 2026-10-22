"""Exact quaternion identity and archived-case check of an anisotropy bound."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sympy as sp
import torch
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rotation(q):
    w,x,y,z=q
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y))).reshape(3,3)
def main():
    w,x,y,z=sp.symbols('w x y z',real=True);v=sp.Matrix([x,y,z]);K=sp.Matrix([[0,-z,y],[z,0,-x],[-y,x,0]])
    R=sp.eye(3)+2*w*K+2*K*K;e=w*w+x*x+y*y+z*z-1
    defect=4*e*((v.dot(v))*sp.eye(3)-v*v.T)
    tests=[sp.expand(a)==0 for a in R.T*R-sp.eye(3)-defect]
    if not all(tests):raise AssertionError('Quaternion identity failed.')
    rows=[];inputs={}
    for name in ('world_preprocess_stress_v1','factor_regular_inputs_v1'):
        p=ROOT/'research/experiments'/name/'results.json';inputs[name]=sha(p);data=json.loads(p.read_text())
        for i,c in enumerate(data['cases']):
            q=torch.tensor(c['normalized_device_quaternion'],dtype=torch.float64)
            s=torch.tensor(c['filtered_scales'],dtype=torch.float64);R=rotation(q);eye=torch.eye(3,dtype=torch.float64)
            Q=R.T@R-eye;delta=float(torch.linalg.matrix_norm(Q,ord=2));a=float(s.max()/s.min())
            P=R@torch.diag(s.pow(-2))@R.T;cov=R@torch.diag(s*s)@R.T
            actual=P@cov-eye
            expanded=R@R.T-eye+R@torch.diag(s.pow(-2))@Q@torch.diag(s*s)@R.T
            expansion_relative=float(torch.linalg.matrix_norm(actual-expanded)/max(float(torch.linalg.matrix_norm(actual)),1e-5))
            actualnorm=float(torch.linalg.matrix_norm(actual,ord=2));bound=delta*(1+(1+delta)*a*a)
            exact_delta=4*abs(float(q@q)-1)*float(q[1:]@q[1:])
            # Roundoff of matrix products is allowed only in numeric validation;
            # the displayed theorem is exact real arithmetic.
            if actualnorm>bound+1e-5:raise AssertionError('Spectral bound failed beyond numerical margin.')
            rows.append({'input':name,'case':i,'axis_ratio':a,'delta':delta,'delta_quaternion_formula':exact_delta,
                'precision_covariance_residual':actualnorm,'upper_bound':bound,'expansion_scaled_roundoff':expansion_relative})
    g=torch.Generator().manual_seed(12089);direct_checks=[]
    for i in range(64):
        # Deliberately nonorthogonal matrices: differentiating the actual factor
        # graph must not require the quaternion unit-norm identity.
        R=(torch.randn((3,3),dtype=torch.float64,generator=g)+torch.eye(3,dtype=torch.float64)).requires_grad_()
        logscale=torch.randn(3,dtype=torch.float64,generator=g,requires_grad=True)
        G=torch.randn((3,3),dtype=torch.float64,generator=g);S=(G+G.T)/2;d=torch.exp(-2*logscale)
        P=R@torch.diag(d)@R.T;gr,gs=torch.autograd.grad((G*P).sum(),(R,logscale))
        er=2*S@R@torch.diag(d);es=-2*d*torch.diagonal(R.T@S@R)
        errors=[float(torch.linalg.vector_norm(a-b)/max(float(torch.linalg.vector_norm(a)),1e-12)) for a,b in ((gr,er),(gs,es))]
        if max(errors)>1e-12:raise AssertionError('Direct factor identity failed.')
        direct_checks.append(max(errors))
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(Path(__file__)),
        'sympy_version':sp.__version__,'symbolic_quaternion_components_pass':tests,'input_sha256':inputs,
        'archived_case_count':len(rows),'numeric_spectral_bound_margin':1e-5,'all_bound_checks_pass':True,'rows':rows,
        'max_precision_covariance_residual':max(x['precision_covariance_residual'] for x in rows),
        'direct_factor_ad_cases':64,'max_direct_factor_ad_error':max(direct_checks),
        'identities':{'quaternion':'R^T R-I=4(||q||^2-1)(||v||^2 I-v v^T)',
            'roundtrip':'P Sigma-I=(R R^T-I)+R D^-2(R^T R-I)D^2 R^T',
            'bound':'||P Sigma-I||_2 <= delta [1+(1+delta) axis_ratio^2]',
            'factor_R_gradient':'2 sym(G) R D^-2','factor_logscale_gradient':'-2 diag(D^-2) * diag(R^T sym(G) R)'},
        'limits':['Exact real-arithmetic identities; numeric checks are corroboration, not the proof.',
            'Upper bound is sufficient, not necessary or a prediction that every high-axis primitive fails.',
            'Does not bound the full renderer gradient, nonlinear gradient error, or training quality.',
            'Does not establish novelty.']}
    out=ROOT/'research/results/precision_roundtrip_defect_v1.json'
    if out.exists():raise ValueError('Preserve existing result.')
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','identities','limits')},indent=2))
if __name__=='__main__':main()
