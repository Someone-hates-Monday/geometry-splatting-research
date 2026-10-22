"""Exact restricted color-field obstruction and chain rule; not a training/PSNR theorem."""
import hashlib,json
from pathlib import Path
import sympy as s
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_sh_isolation_v12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);target=OUT/'math_results.json';assert not target.exists();native=ROOT/'sources/tgs_native_diagnostic_v1/native';forward=native/'cuda_rasterizer/forward.cu';glm=native/'third_party/glm/glm/detail/type_vec3.inl';old='(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)';assert old in forward.read_text();assert ': x(scalar), y(scalar), z(scalar)' in glm.read_text()
    k=s.Rational(1,4);p=s.Matrix([1,0,0]);q=s.Matrix([p[2]]*3);d=s.Matrix([0,0,1]);radii=[s.Integer(1),s.Integer(2)];rows=[];values=[]
    for radius in radii:
        c=p-radius*d;new=p-c;new=new/s.sqrt(new.dot(new));legacy=q-c;legacy=legacy/s.sqrt(legacy.dot(legacy));assert new==d;value=s.Rational(1,2)-k*legacy[0];values.append(value);assert value>s.Rational(1,2) and value<1;rows.append(dict(camera=list(map(str,c)),corrected_direction=list(map(str,new)),legacy_direction=list(map(str,legacy)),legacy_color=str(value),legacy_color_decimal=str(value.evalf(40))))
    delta=s.simplify(values[0]-values[1]);assert delta.is_positive;lower=s.simplify(delta/2);mid=s.simplify(sum(values)/2);assert s.simplify(values[0]-mid-lower)==0 and s.simplify(mid-values[1]-lower)==0
    x,y,z,tx,ty,tz=s.symbols('x y z tx ty tz',real=True);pv=s.Matrix([x,y,z]);tv=s.Matrix([tx,ty,tz]);qv=s.Matrix([z,z,z]);J=qv.jacobian(pv);assert J==s.Matrix([[0,0,1],[0,0,1],[0,0,1]]);translation=s.simplify(s.Matrix([z+tz]*3)-(pv+tv)-(qv-pv));assert translation==s.Matrix([tz-tx,tz-ty,0])
    a,b,c=s.symbols('a b c',real=True);u=s.Matrix([a-1,b,c+1]);F=s.Rational(1,2)-k*u[0]/s.sqrt(u.dot(u));gq=s.Matrix([s.diff(F,v) for v in [a,b,c]]);reported=gq.subs({a:z,b:z,c:z});true=s.Matrix([s.diff(F.subs({a:z,b:z,c:z}),v) for v in [x,y,z]]);assert all(s.simplify(v)==0 for v in true-J.T*reported);point={x:1,y:0,z:0};rg=s.simplify(reported.subs(point));tg=s.simplify(true.subs(point));assert rg[0]!=0 and tg[0]==0
    curl_entry=s.simplify((s.diff(reported[0],z)-s.diff(reported[2],x)).subs(point));assert curl_entry!=0
    out=dict(exact_checks_pass=True,sympy=s.__version__,counterexample=rows,color_gap=str(delta),color_gap_decimal=str(delta.evalf(40)),best_two_sample_uniform_color_error_lower_bound=str(lower),lower_bound_decimal=str(lower.evalf(40)),achieving_common_color=str(mid),legacy_center_Jacobian=str(J),common_translation_residual=str(translation),assumed_reported_color_gradient_at_point=list(map(str,rg)),chain_correct_gradient_at_point=list(map(str,tg)),mixed_partial_disagreement=str(curl_entry),source_forward_sha256=sha(forward),source_GLM_sha256=sha(glm),script_sha256=sha(Path(__file__)),scope='Real arithmetic, one admissible degree1 color field; same corrected direction at two different camera distances. No coefficient-only direction function can match both. Chain/curl claim concerns the isolated assumed SH-color component, not complete renderer/training.',novelty_established=False,actual_PSNR_cause_established=False,training=False)
    target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if __name__=='__main__':main()
