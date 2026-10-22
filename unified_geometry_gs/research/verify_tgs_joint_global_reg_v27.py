"""Exact symbolic source-conditioned regularizer corrections;no GPU/model execution."""
import json
from pathlib import Path
import sympy as sp
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_joint_edit_common_v27 import OUT

def main():
    source=ROOT/'sources/tgs_subdivision_static_v1/triangle-splatting';paths=[source/'train.py',source/'utils/loss_utils.py',ROOT/'TGS_JOINT_GLOBAL_REG_ADDENDUM_V27.md',Path(__file__)]
    write_new(OUT/'global_reg_plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},exact_symbolic=True,data_informed_source_scope_addendum=True,training=False,new_native_forwards=0))
    assert 'torch.abs(triangles.get_opacity).mean() * args.lambda_opacity' in (source/'train.py').read_text()
    N,k=sp.symbols('N k',integer=True,positive=True);S,Oi,Oj,Z,dS,dN,A=sp.symbols('S Oi Oj Z dS dN A',real=True);lam=sp.symbols('lambda',nonnegative=True);mu=S/N
    pair=lam*((S+3*(Oi+Oj))/(N+6)-S/N);unary=lam*((S+3*Oi)/(N+3)-S/N)+lam*((S+3*Oj)/(N+3)-S/N)
    pair_eta=-9*lam*(Oi+Oj-2*mu)/((N+6)*(N+3));assert sp.cancel(pair-unary-pair_eta)==0
    many=lam*((S+3*Z)/(N+3*k)-S/N);many_formula=3*lam*(Z-k*mu)/(N+3*k);assert sp.cancel(many-many_formula)==0
    many_eta=-9*lam*(k-1)*(Z-k*mu)/((N+3*k)*(N+3));assert sp.cancel(many-3*lam*(Z-k*mu)/(N+3)-many_eta)==0
    assert sp.cancel(lam*((S+dS)/(N+dN)-S/N)-lam*(dS-mu*dN)/(N+dN))==0
    assert sp.cancel((N+3*k)/A-N/A-k*((N+3)/A-N/A))==0
    # Feasible toy:10 opacities sum4;selected .8 and .7;remaining sum2.5.
    values={N:10,S:4,Oi:sp.Rational(4,5),Oj:sp.Rational(7,10),lam:sp.Rational(1,10)}
    exact_eta=sp.cancel(pair_eta.subs(values));assert exact_eta==sp.Rational(-63,20800)
    write_new(OUT/'global_reg_results.json',dict(all_identities_passed=True,sympy=sp.__version__,pair_interaction=str(pair_eta),k_edit_change=str(many_formula),k_edit_interaction=str(many_eta),
        toy_pair_interaction_exact=str(exact_eta),toy_not_real_model=True,assumptions='N>0;denominators nonzero;area A>0 when used;distinct inherited-opacity edits;bound additionally0<=opacity<=1,k<=N',
        not_FP32_training_reduction_proof=True,training=False,new_native_forwards=0,plan_sha256=sha(OUT/'global_reg_plan.json')))
    print(json.dumps(dict(all_identities_passed=True,toy_exact_eta=str(exact_eta),sympy=sp.__version__)),flush=True)

if __name__=='__main__':main()
