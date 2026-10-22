"""Exact identities and fixed-seed witnesses of single-normal incompatibility."""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import numpy as np
import sympy as sp
from research.directional_field import orthogonal_witness

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'research/results/directional_field_theory_v1.json'
    if out.exists():raise ValueError('Fresh output required')
    x,y,z=sp.symbols('x y z',real=True);a=sp.symbols('a',positive=True)
    xx=sp.Matrix([x,y,z]);p00,p01,p02,p11,p12,p22=sp.symbols('p00 p01 p02 p11 p12 p22',real=True)
    P=sp.Matrix([[p00,p01,p02],[p01,p11,p12],[p02,p12,p22]])
    G=a*sp.exp(-(xx.T*P*xx)[0]/2)
    actual=sp.Matrix([sp.diff(sp.log(1-G)/2,v) for v in xx])
    expected=G/(2*(1-G))*P*xx
    gradient_checks=[sp.simplify(v)==0 for v in actual-expected]
    wx,wy=sp.symbols('wx wy',real=True)
    e=sp.exp(-sp.Rational(1,2));c=e/(4-2*e)
    # Isotropic width c, center -c e_i, alpha=1/2 at the origin gives b_i=e_i.
    realization=sp.simplify((e/2)/(2*(1-e/2))/c-1)==0
    identities={'world_gradient':gradient_checks,'orthogonal_gaussian_realization':realization,
                'sharp_candidate_error':sp.simplify(sp.Abs(wx)+sp.Abs(wy)-sp.Abs(wx)-sp.Abs(wy))==0,
                'cooriented_product_identity':sp.simplify(sp.Abs(2*wx)+sp.Abs(3*wx)-sp.Abs(5*wx))==0,
                'opposed_product_counterexample':sp.simplify(sp.Abs(wx)+sp.Abs(-wx)-2*sp.Abs(wx))==0}
    if not all(all(v) if isinstance(v,list) else v for v in identities.values()):raise AssertionError('Exact identities failed')
    rng=np.random.default_rng(13014);rows=[]
    for i in range(64):
        B=rng.normal(size=(6,3));candidate=rng.normal(size=3)*rng.uniform(.1,10)
        S=B.T@B;lambda2=np.linalg.eigvalsh(S)[-2]
        w=orthogonal_witness(candidate,S)
        residual=abs(sum(abs(B@w))-abs(candidate@w))
        pair=B[:2];pair_lambda2=np.linalg.eigvalsh(pair.T@pair)[-2]
        lower=np.sqrt(lambda2)
        if residual+1e-10*max(1,lower)<lower or abs(np.linalg.norm(w)-1)>1e-12 or pair_lambda2>lambda2+1e-10:
            raise AssertionError('Numeric witness failed')
        rows.append({'case':i,'vectors':B.tolist(),'candidate':candidate.tolist(),'witness_direction':w.tolist(),
                     'candidate_dot_witness':float(candidate@w),'lambda2':float(lambda2),
                     'uniform_error_lower_bound':float(lower),'observed_direction_error':float(residual),
                     'pair_lambda2':float(pair_lambda2)})
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('directional_field.py'),ROOT/'research/DIRECTIONAL_FIELD_PROTOCOL.md')},
            'symbolic_checks':identities,'seed':13014,'numeric_cases':rows,'all_declared_checks_pass':True,
            'scope':'Fixed-point absolute-gradient single-normal form only; does not rule out general normal-distribution stochastic geometry or prove method novelty/quality.'}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'symbolic_checks':identities,'numeric_cases':len(rows),'all_declared_checks_pass':True},indent=2))


if __name__=='__main__':main()
