"""Exact checks for the post-hoc distance-transfer mechanism, no model or GPU evaluation."""
import hashlib,json
from pathlib import Path
import sympy as s
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_sh_shell_v13'
t,k=s.symbols('t k',real=True,positive=True)

def main():
    target=OUT/'radius_asymptotics.json';assert not target.exists()
    G=(1-t*t)*s.diff(s.legendre(3,t),t)
    low=s.Rational(12,7)*s.legendre(2,t)
    high=-s.Rational(12,7)*s.legendre(4,t)
    assert s.expand(G-low-high)==0
    inner=lambda x,y:s.integrate(s.expand(x*y),(t,-1,1))/2
    assert inner(low,high)==0
    eta=s.simplify(inner(low,low)/inner(G,G));assert eta==s.Rational(9,14)
    ratio2=s.simplify(inner((k-1)*low-high,(k-1)*low-high)/inner(G,G))
    assert s.simplify(ratio2-((k-1)**2*eta+1-eta))==0
    checks=[]
    for kk in [s.Rational(1,2),s.Integer(1),s.Integer(2),s.Integer(4)]:
        checks.append(dict(radius_factor=str(kk),zonal_P3_frozen_L3_ratio=str(s.sqrt(ratio2.subs(k,kk))),
                           any_nonzero_first_order_frozen_L4_ratio=str(abs(kk-1))))
    target.write_text(json.dumps(dict(exact_checks_pass=True,zonal_P3_eta=str(eta),
        exact_zonal_P3_ratio_squared=str(ratio2),checks=checks,sympy=s.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        posthoc_mechanism=True,training=False),indent=2)+'\n')
    print(json.dumps(dict(exact_checks_pass=True,zonal_P3_eta=str(eta))),flush=True)

if __name__=='__main__':main()
