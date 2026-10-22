"""Exact family showing the axis-ratio squared bound has sharp order."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sympy as sp
import torch
ROOT=Path(__file__).resolve().parents[1]
def main():
    e,a=sp.symbols('epsilon a',positive=True)
    R=sp.Matrix([[-e,1+e,0],[1+e,-e,0],[0,0,-1-2*e]])
    D2=sp.diag(1,a*a,1);P=R*D2.inv()*R.T;cov=R*D2*R.T;delta=4*e*(1+e)
    E=P*cov-sp.eye(3);limit=E.applyfunc(lambda v:sp.limit(v/(delta*a*a),a,sp.oo))
    expected=-R[:,0]*R[:,1].T/2
    if any(sp.simplify(x)!=0 for x in limit-expected):raise AssertionError('Leading rank-one limit failed.')
    coefficient=sp.simplify((R[:,0].dot(R[:,0]))/2)
    if coefficient!=e*e+e+sp.Rational(1,2):raise AssertionError('Rank-one singular norm identity failed.')
    # A different derivation from the actual quaternion polynomial.
    v=sp.Matrix([sp.sqrt((1+e)/2),sp.sqrt((1+e)/2),0]);K=sp.Matrix([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if any(sp.simplify(x)!=0 for x in R-(sp.eye(3)+2*K*K)):raise AssertionError('Quaternion family identity failed.')
    rows=[];ev=2.**-24
    Rt=torch.tensor([[-ev,1+ev,0],[1+ev,-ev,0],[0,0,-1-2*ev]],dtype=torch.float64)
    dv=4*ev*(1+ev);target=.5+ev+ev*ev
    for av in (10.,100.,1000.,10000.,100000.,1000000.):
        ds=torch.diag(torch.tensor([1.,av*av,1.],dtype=torch.float64));pv=Rt@torch.linalg.inv(ds)@Rt.T;cv=Rt@ds@Rt.T
        value=float(torch.linalg.matrix_norm(pv@cv-torch.eye(3,dtype=torch.float64),ord=2))/(dv*av*av)
        rows.append({'axis_ratio':av,'scaled_residual':value,'limit':target,'absolute_difference':abs(value-target)})
    if rows[-1]['absolute_difference']>1e-9:raise AssertionError('Numerical corroboration did not approach exact limit.')
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'exact_quaternion_family_pass':True,'exact_rank_one_limit_pass':True,'exact_limit_norm':str(coefficient),
        'quaternion_family':'q=(0,sqrt((1+epsilon)/2),sqrt((1+epsilon)/2),0)',
        'scale_family':'D=diag(1,a,1), epsilon>0 fixed, a->infinity',
        'identity':'lim ||P Sigma-I||_2/(delta a^2)=1/2+epsilon+epsilon^2',
        'numeric_epsilon':ev,'numeric_rows':rows,
        'consequence':'No axis-ratio-independent finite C can bound this residual by C delta on the stated family.',
        'limits':['Exact real-arithmetic matrix residual result, not a lower bound for every primitive or any full-renderer gradient.',
            'Not proof that every family member is produced by a float32 quaternion-normalization kernel.',
            'Does not establish reconstruction improvement or novelty.']}
    out=ROOT/'research/results/roundtrip_sharpness_v1.json'
    if out.exists():raise ValueError('Keep existing result.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
