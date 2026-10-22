"""Independent rational certificate and saved causal-intervention audit."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
from tgs_mixed_numpy import evaluate,complex_gradient

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_radius_contraction_v1'
TILE=OUT/'tile_coverage'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    certificate=json.loads((TILE/'rational_certificate.json').read_text())
    q=[[F(x),F(y)] for x,y in certificate['vertices']]
    r=F(3,2);s=(q[0][0]+r,q[0][1]+r);u0=F(1,50)
    ymax=s[1]+(1-u0)*(q[2][1]-s[1])
    assert ymax==F(certificate['inner_support_ymax'])==F(3211,25)
    assert (ymax+15)//16==8 and ymax//16+1==9
    n=(F(3,5),F(4,5));b=-sum(n[j]*q[1][j] for j in range(2))
    phi=n[0]*40+n[1]*128+b
    assert phi==F(-7,40) and F(1,2)*(-phi/r)==F(7,120)>F(1,100)>F(1,255)
    before=json.loads((TILE/'native_reproduction.json').read_text())
    after=json.loads((TILE/'upper_fixed_reproduction.json').read_text())
    assert np.max(before['native_samples'][0])==0 and np.min(after['native_samples'][0])>.05
    assert after['forward_max_abs']<3e-5 and after['gradient_relative']<1e-3
    with np.load(TILE/'upper_fixed_reproduction.npz') as z:
        f={k:z[k].copy() for k in z.files if k not in ['native_rgb','native_aux','native_gradient','reference_gradient']}
        g=z['native_gradient'];saved=z['reference_gradient']
        assert z['native_aux'][1,128,40]>.05
    cs=complex_gradient(f['theta'],f)
    assert np.linalg.norm(cs-saved)<1e-8
    assert np.linalg.norm(g-cs)/np.linalg.norm(cs)<1e-3
    build=json.loads((TILE/'upper_fix_build/build_receipt.json').read_text())
    assert sha(TILE/'upper_fix_build/tgs_diag_6d61f4c_tile_upper_v1.so')==build['module_sha256']==after['module_sha256']
    original=ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/generated'
    assert (TILE/'upper_fix_build/generated/backward.cu').read_bytes()==(original/'backward.cu').read_bytes()
    expected=(original/'forward.cu').read_text()
    for axis,block in [('x','BLOCK_X'),('y','BLOCK_Y')]:
        old=f'uint b{axis}1 = min(grid.{axis}, max(0, (uint)((intersect_{axis} + {block} - 1) / {block})));'
        new=f'uint b{axis}1 = (uint)fminf((float)grid.{axis}, fmaxf(0.0f, floorf(intersect_{axis} / {block}) + 1.0f));'
        assert expected.count(old)==2;expected=expected.replace(old,new)
    assert expected==(TILE/'upper_fix_build/generated/forward.cu').read_text()
    plan=json.loads((OUT/'input_plan.json').read_text());entry=next(e for e in plan['stress_cases'] if e['name']=='stress_40')
    result=json.loads((TILE/'stress40_upper_only.json').read_text())
    assert result['input_sha256']==entry['sha256']==sha(ROOT/entry['input'])
    with np.load(ROOT/entry['input']) as z:stress={k:z[k].copy() for k in z.files}
    with np.load(TILE/'stress40_upper_only.npz') as z:
        gradients=z['native_gradients'];reference=z['reference_gradients'];samples=z['after_samples']
        assert np.array_equal(z['before_samples'],result['before_samples'])
        assert np.array_equal(samples,result['after_samples'])
    for pi,p in enumerate(result['rows']):
        stress['pixel_grad']=stress['patterns'][pi]
        cs=complex_gradient(stress['theta'],stress);_,rgb,_=evaluate(stress['theta'].astype(float),stress)
        assert np.linalg.norm(cs-reference[pi])<1e-7
        ge=np.linalg.norm(gradients[pi,:9]-cs[:9])/np.linalg.norm(cs[:9])
        se=np.linalg.norm(gradients[pi,9:]-cs[9:])/np.linalg.norm(cs[9:])
        fe=np.max(np.abs(samples-rgb))
        assert ge<=1e-3 and se<=1e-3 and fe<=3e-5 and p['passes_same_numeric_gates']
        assert abs(ge-p['geometry_relative'])<1e-9 and abs(se-p['scalar_relative'])<1e-9
    real=json.loads((TILE/'real_upper_only.json').read_text())
    real_plan=json.loads((TILE/'real_upper_only_plan.json').read_text())
    assert sha(TILE/'real_upper_only_plan.json')==real['plan_sha256']
    assert sha(ROOT/'measure_tgs_tile_upper_real.py')==real_plan['script_sha256']
    real_rows=[]
    for row in real['rows']:
        ci=row['camera']
        with np.load(TILE/f'real_camera_{ci}_upper_only.npz') as z:
            a=z['before_rgb'];b=z['after_rgb'];aa=z['before_alpha'];ab=z['after_alpha']
            ra=z['before_radii'];rb=z['after_radii']
        d=np.abs(b.astype(float)-a.astype(float));ad=ab.astype(float)-aa.astype(float)
        assert abs(d.max()-row['rgb_max_abs'])<1e-15 and abs(d.mean()-row['rgb_mean_abs'])<1e-15
        assert int((d.max(0)>0).sum())==row['pixels_changed_any']
        assert int((d.max(0)>3e-5).sum())==row['pixels_changed_above_3e_5']
        assert int(((ra==0)&(rb>0)).sum())==row['newly_visible']
        assert int(((ra>0)&(rb==0)).sum())==row['lost_visibility']
        assert int((ad>3e-5).sum())==row['alpha_positive_pixels']
        assert int((ad<-3e-5).sum())==row['alpha_negative_pixels']
        assert ad.max()==row['alpha_max_change'] and ad.min()==row['alpha_min_change']
        historical=ROOT/f'experiments/tgs_real_event_v1/baseline_{ci+2}.npz'
        assert sha(historical)==row['historical_baseline_sha256']
        with np.load(historical) as z:assert np.array_equal(a,z['fixed_rgb'])
        real_rows.append(dict(image=row['image'],changed_pixels_above_3e_5=row['pixels_changed_above_3e_5'],
                             rgb_max_change=row['rgb_max_abs'],rgb_mean_change=row['rgb_mean_abs'],newly_visible=row['newly_visible']))
    summary=dict(evidence_audit_passed=True,rational_alpha='7/120',legacy_upper=8,correct_upper=9,
        native_missing_pixel_recovered=True,rational_gradient_relative=after['gradient_relative'],
        frozen_stress40_max_geometry_relative=result['max_geometry_relative'],
        upper_bound_only_causal_intervention_verified=True,contraction_v1_still_rejected=True,training=False)
    summary['real_full_scene_causal_changes']=real_rows
    summary['image_quality_gain_established']=False
    (TILE/'summary_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
