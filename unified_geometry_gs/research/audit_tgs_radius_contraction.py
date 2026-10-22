"""Independently recompute all frozen VJP/FD gates and selected-pixel support.

The original window/radii branch flag did not check tile membership. Infer
native sample support for single-triangle stresses without changing raw logs.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from tgs_mixed_numpy import evaluate as rgb_ref,complex_gradient as rgb_grad
from tgs_aux_numpy import evaluate as aux_ref,complex_gradient as aux_grad

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_radius_contraction_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(g,ref):
    norm=np.linalg.norm(ref)
    return bool(np.linalg.norm(g-ref)/norm<=1e-3) if norm>1e-10 else bool(np.max(np.abs(g-ref))<=1e-6)


def main():
    plan=json.loads((OUT/'input_plan.json').read_text());data=json.loads((OUT/'results.json').read_text())
    run=json.loads((OUT/'execution_plan.json').read_text())
    assert sha(OUT/'input_plan.json')==data['input_plan_sha256']==run['input_plan_sha256']
    assert sha(OUT/'execution_plan.json')==data['execution_plan_sha256']
    assert sha(ROOT/'probe_tgs_radius_contraction.py')==run['script_sha256']
    assert sha(ROOT/'TGS_RADIUS_CONTRACTION_PROTOCOL_V1.md')==plan['protocol_sha256']
    modules={'contraction':OUT/'build/tgs_diag_6d61f4c_contraction_v1.so',
             'piecewise':ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/tgs_diag_6d61f4c_piecewise_v1.so'}
    for name,path in modules.items():assert sha(path)==run['module_sha256'][name]
    summary_rows=[];fd_count=0;support_changes=[]
    for family,entries in [('old',plan['old_cases']),('stress',plan['stress_cases'])]:
        for entry in entries:
            path=ROOT/entry['input'];assert sha(path)==entry['sha256']
            with np.load(path) as z:f={k:z[k].copy() for k in z.files}
            for k in ['n','w','h','degree']:f[k]=int(f[k])
            for k in ['name','kind','check']:f[k]=str(f[k])
            ref=aux_ref if f['kind']=='aux' else rgb_ref;gradient=aux_grad if f['kind']=='aux' else rgb_grad
            patterns=f['patterns'] if family=='stress' else [f['pixel_grad']]
            variants={r['variant']:r for r in data['records'] if r['name']==f['name']}
            images={};aux_images={};diagnostics={}
            for variant,row in variants.items():
                gfile=OUT/(variant+'_'+f['name']+'.npz')
                with np.load(gfile) as z:
                    images[variant]=z['native_rgb'].copy();aux_images[variant]=z['native_aux'].copy()
                    native_samples=z['native_samples'].copy();gs=z['native_gradients'].copy();rs=z['reference_gradients'].copy()
                group_fail=[];forward_fail=[];predicted=[]
                for pi,pattern in enumerate(patterns):
                    f['pixel_grad']=pattern;p=row['patterns'][pi]
                    loss,samples,fp=ref(f['theta'].astype(float),f);cs=gradient(f['theta'],f)
                    exact=np.array(p['reference_gradient']);g=np.array(p['native_gradient']);n=f['n']
                    assert np.array_equal(g,gs[pi]) and np.array_equal(exact,rs[pi])
                    assert np.linalg.norm(cs-exact)<=1e-8*max(np.linalg.norm(exact),1)
                    assert fp==p['metadata']['fingerprint'] and abs(loss-p['reference_loss'])<1e-9
                    fe=float(np.max(np.abs(native_samples-samples)));assert abs(fe-p['forward_max_abs'])<1e-9
                    if f['check']=='homothety_nullspace':gradient_gate=abs(g@f['special_direction'])<=1e-6
                    else:gradient_gate=check(g[:9*n],cs[:9*n]) and check(g[9*n:11*n],cs[9*n:11*n])
                    if f['degree']>=0:gradient_gate=gradient_gate and check(g[11*n:],cs[11*n:])
                    if not gradient_gate:group_fail.append(pi)
                    if fe>3e-5:forward_fail.append(pi)
                    predicted.append(bool(gradient_gate and fe<=3e-5))
                native_support=(np.abs(native_samples-f['background']).max(-1)>1e-7) if family=='stress' else None
                native_switches=0
                for item in row['finite_differences']:
                    pi=item['pattern'];f['pixel_grad']=patterns[pi]
                    h=item['step'];di=item['direction'];d=f['fd_directions'][pi,di]
                    plus=(f['theta'].astype(float)+h*d).astype(np.float32)
                    minus=(f['theta'].astype(float)-h*d).astype(np.float32)
                    actual=(plus.astype(float)-minus.astype(float))/(2*h)
                    assert np.array_equal(actual,item['actual_direction'])
                    lp,_,fpp=ref(plus.astype(float),f);lm,_,fpm=ref(minus.astype(float),f)
                    assert abs(lp-item['cpu_plus_loss'])<1e-9 and abs(lm-item['cpu_minus_loss'])<1e-9
                    assert fpp==item['plus_fp'] and fpm==item['minus_fp'] and item['smooth']
                    cs=rs[pi]
                    assert abs(cs@actual-item['expected'])<1e-7*max(abs(item['expected']),1)
                    assert abs((item['native_plus_loss']-item['native_minus_loss'])/(2*h)-item['native_fd'])<1e-10
                    for side in ['plus','minus']:
                        samples=np.array(item[side+'_samples'])
                        assert abs((samples*patterns[pi]).sum()-item['native_'+side+'_loss'])<1e-10
                    if native_support is not None:
                        plus_support=np.abs(np.array(item['plus_samples'])-f['background']).max(-1)>1e-7
                        minus_support=np.abs(np.array(item['minus_samples'])-f['background']).max(-1)>1e-7
                        native_switches+=int(not(np.array_equal(plus_support,native_support) and np.array_equal(minus_support,native_support)))
                    fd_count+=1
                if family=='old':
                    p=ROOT/entry['old_output'];assert sha(p)==entry['old_output_sha256']
                    with np.load(p) as z:assert np.array_equal(images[variant],z['native_rgb'])
                assert all(predicted)==row['gate']
                diagnostics[variant]=dict(gate=row['gate'],gradient_failed_patterns=group_fail,forward_tolerance_failed_patterns=forward_fail,
                    native_selected_pixel_support_switches=native_switches,
                    maximum_geometry_relative=max(p['groups']['geometry']['relative'] for p in row['patterns']),
                    maximum_scalar_relative=max(p['groups']['sigma_opacity']['relative'] for p in row['patterns']),
                    forward_max_abs=max(p['forward_max_abs'] for p in row['patterns']))
                if native_switches:support_changes.append(dict(name=f['name'],variant=variant,pairs=native_switches))
            assert np.array_equal(images['piecewise'],images['contraction']) and np.array_equal(aux_images['piecewise'],aux_images['contraction'])
            summary_rows.append(dict(name=f['name'],family=family,variants=diagnostics,forward_images_identical=True))
    failures=[r['name'] for r in summary_rows if not r['variants']['contraction']['gate']]
    gradient_failures=[r['name'] for r in summary_rows if r['variants']['contraction']['gradient_failed_patterns']]
    assert failures==data['contraction_failed_cases'] and fd_count==4608
    assert not data['performance_measurement_allowed']
    summary=dict(evidence_audit_passed=True,rows=summary_rows,old_case_count=18,stress_case_count=48,
        stress_vjps_per_case=8,finite_difference_probe_step_pairs=fd_count,
        contraction_failed_cases=failures,contraction_gradient_failed_cases=gradient_failures,
        supplementary_native_support_switches=support_changes,
        original_branch_flag_scope='CPU window/contributor fingerprint and native radii only; it did not verify selected-pixel tile membership',
        performance_measurement_allowed=False,training=False)
    (OUT/'summary_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['evidence_audit_passed','finite_difference_probe_step_pairs',
        'contraction_failed_cases','contraction_gradient_failed_cases','supplementary_native_support_switches','performance_measurement_allowed']}))


if __name__=='__main__':main()
