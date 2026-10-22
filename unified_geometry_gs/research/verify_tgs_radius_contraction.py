"""Exact symbolic checks; no claim about float32 error or scientific novelty."""
import hashlib
import json
from pathlib import Path
import sympy as s

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_radius_contraction_v1'


def main():
    nx,ny,ax,ay,cx,cy,px,py,beta=s.symbols('nx ny ax ay cx cy px py beta',real=True)
    n=s.Matrix([nx,ny]);a=s.Matrix([ax,ay]);c=s.Matrix([cx,cy]);p=s.Matrix([px,py])
    G=beta*p;Gc=beta*(p-c);B=beta
    global_gamma=-(n.dot(G)-n.dot(a)*B)
    local_gamma=-(n.dot(Gc)+n.dot(c-a)*B)
    assert s.expand(global_gamma-local_gamma)==0
    normal_chain=G-a*B;local_chain=Gc+(c-a)*B
    assert (normal_chain-local_chain).applyfunc(s.expand)==s.zeros(2,1)
    for sign in [-1,1]:
        assert s.expand((sign*n).dot(sign*Gc)+(sign*n).dot(c-a)*(sign*B)+local_gamma)==0
    dnx,dny,dcx,dcy,db=s.symbols('dnx dny dcx dcy db',real=True)
    dn=s.Matrix([dnx,dny]);dc=s.Matrix([dcx,dcy])
    local_dphi=dn.dot(p-c)-n.dot(dc)+db+dn.dot(c)+n.dot(dc)
    assert s.expand(local_dphi-(dn.dot(p)+db))==0
    w,ell,sigma,alpha=s.symbols('w ell sigma alpha',real=True)
    assert s.expand(-(w+(1-w))*ell*sigma*alpha+ell*sigma*alpha)==0
    result=dict(passed=True,sympy=s.__version__,checks=['global/local Gamma identity','normal cotangent identity',
        'consistent orientation reversal','moving-anchor cancellation','unit-mass active-edge weights'],
        assumptions=['first-order VJP on fixed contributor/visibility branches','max weights sum to one',
                     'local alpha chain factor is included in beta','valid contributors have nonzero phi'],
        float32_accuracy_proven=False,novelty_established=False,
        protocol_sha256=hashlib.sha256((ROOT/'TGS_RADIUS_CONTRACTION_PROTOCOL_V1.md').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'symbolic_verification.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
