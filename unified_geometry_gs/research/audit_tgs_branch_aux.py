"""Independent NumPy/complex-step and saved finite-difference evidence audit."""
import hashlib
import json
from pathlib import Path
import numpy as np

from tgs_mixed_numpy import evaluate as rgb_evaluate,complex_gradient as rgb_gradient
from tgs_aux_numpy import evaluate as aux_evaluate,complex_gradient as aux_gradient

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_branch_aux_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def main():
    old_files=0
    for name in ['tgs_backward_v1','tgs_mixed_backward_v1']:
        inventory=ROOT/'experiments'/name/'evidence_manifest.json'
        for e in json.loads(inventory.read_text())['files']:
            assert sha(ROOT/e['path'])==e['sha256'];old_files+=1
    plan=json.loads((OUT/'protocol.json').read_text())
    assert sha(ROOT/'probe_tgs_branch_aux.py')==plan['script_sha256']
    results={v:json.loads((OUT/(v+'_results.json')).read_text()) for v in ['geometry_fixed','piecewise']}
    receipt=json.loads((OUT/'piecewise_build/build_receipt.json').read_text())
    module=OUT/'piecewise_build/tgs_diag_6d61f4c_piecewise_v1.so'
    assert sha(module)==receipt['module_sha256']==results['piecewise']['module_sha256']
    assert sha(ROOT/'tgs_piecewise_backend.py')==receipt['generator_sha256']
    for e in receipt['sources']:
        prefix='/mnt/c/Users/Someone/Desktop/SuGaR-main/'
        p=ROOT.parents[1]/e['path'][len(prefix):]
        assert sha(p)==e['sha256']
    rows=[];pairs=0
    for i,scene in enumerate(plan['scenes']):
        name=scene['name'];path=OUT/(name+'_input.npz');assert sha(path)==scene['input_sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for key in ['n','w','h','degree']:f[key]=int(f[key])
        for key in ['name','kind','check']:f[key]=str(f[key])
        evaluate=aux_evaluate if f['kind']=='aux' else rgb_evaluate
        gradient=aux_gradient if f['kind']=='aux' else rgb_gradient
        loss,samples,fp=evaluate(f['theta'].astype(float),f);cs=gradient(f['theta'],f)
        assert fp==scene['metadata']['fingerprint']
        images={};aux_images={};colors={};summaries={}
        for variant in ['geometry_fixed','piecewise']:
            row=results[variant]['rows'][i];assert row['name']==name
            exact=np.array(row['exact_gradient']);g=np.array(row['native_gradient']);n=f['n']
            # Absolute bound also covers the saturated all-zero reference.
            cs_error=rel(cs,exact)
            assert np.max(np.abs(cs-exact))<1e-8*max(np.max(np.abs(exact)),1)
            assert abs(loss-row['cpu_loss'])<1e-11 and row['metadata']['fingerprint']==fp
            with np.load(OUT/(variant+'_'+name+'.npz')) as z:
                packed=np.r_[z['native_grad_0'].ravel(),z['native_grad_1'].ravel(),z['native_grad_3'].ravel()]
                assert np.array_equal(packed,g)
                assert np.max(np.abs(z['reference_samples']-samples))<1e-11
                weights=f['aux_grad'] if f['kind']=='aux' else f['pixel_grad']
                assert abs((z['native_samples'].astype(float)*weights).sum()-row['native_loss'])<1e-12
                forward_error=float(np.max(np.abs(z['native_samples']-samples)))
                assert abs(forward_error-row['forward_max_abs'])<1e-11
                images[variant]=z['native_rgb'].copy();aux_images[variant]=z['native_aux'].copy()
                colors[variant]=z['native_grad_2'].copy()
            if f['check']=='homothety_nullspace':ds=np.array([f['special_direction']])
            else:ds=np.eye(len(exact))
            fd_stats=[]
            for step in row['finite_differences']:
                cpu=[];gpu=[];expected=[]
                for j,item in enumerate(step['items']):
                    h=step['step'];d=ds[j]
                    plus=(f['theta'].astype(float)+h*d).astype(np.float32)
                    minus=(f['theta'].astype(float)-h*d).astype(np.float32)
                    lp,_,fpp=evaluate(plus.astype(float),f);lm,_,fpm=evaluate(minus.astype(float),f)
                    assert fpp==fpm==fp and item['smooth']
                    assert abs(lp-item['cpu_plus_loss'])<1e-11 and abs(lm-item['cpu_minus_loss'])<1e-11
                    actual=(plus.astype(float)-minus.astype(float))/(2*h)
                    linearized=float(cs@actual)
                    assert abs(linearized-item['expected_linearized'])<1e-7*max(abs(linearized),1)
                    native_fd=(item['native_plus_loss']-item['native_minus_loss'])/(2*h)
                    assert abs(native_fd-item['native_fd'])<1e-12
                    assert abs((lp-lm)/(2*h)-item['cpu_fd'])<3e-5*max(abs(item['cpu_fd']),1)
                    for side in ['plus','minus']:
                        assert abs((np.array(item[side+'_samples'])*weights).sum()-item['native_'+side+'_loss'])<1e-12
                    cpu.append((lp-lm)/(2*h));gpu.append(native_fd);expected.append(linearized);pairs+=1
                expected=np.array(expected)
                fd_stats.append(dict(step=h,cpu_fd_error_l2=float(np.linalg.norm(np.array(cpu)-expected)),
                                     native_fd_error_l2=float(np.linalg.norm(np.array(gpu)-expected)),
                                     reference_directional_l2=float(np.linalg.norm(expected))))
            if f['check']=='homothety_nullspace':
                native_direction=float(g@f['special_direction']);true_direction=float(cs@f['special_direction'])
                assert abs(true_direction)<1e-12
                assert abs(native_direction-row['native_direction'])<1e-12
                gate=abs(native_direction)<1e-6
            else:
                native_direction=true_direction=None
                a=rel(g[:9*n],cs[:9*n])<1e-3 if np.linalg.norm(cs[:9*n])>1e-10 else np.max(np.abs(g[:9*n]))<1e-6
                b=rel(g[9*n:],cs[9*n:])<1e-3 if np.linalg.norm(cs[9*n:])>1e-10 else np.max(np.abs(g[9*n:]))<1e-6
                gate=a and b and forward_error<3e-5 and row['all_perturbations_smooth']
            assert bool(gate)==row['gate']
            summaries[variant]=dict(gate=bool(gate),geometry_relative_error=rel(g[:9*n],cs[:9*n]),
                gradient_absolute_error_max=float(np.max(np.abs(g-cs))),reference_max_abs=float(np.max(np.abs(cs))),
                native_direction=native_direction,reference_direction=true_direction,complex_step_reference_error=cs_error,
                finite_difference_statistics=fd_stats)
        assert np.array_equal(images['geometry_fixed'],images['piecewise'])
        assert np.array_equal(aux_images['geometry_fixed'],aux_images['piecewise'])
        if f['check']=='saturated_zero':
            assert np.max(np.abs(cs))==0
            assert np.array_equal(colors['geometry_fixed'],colors['piecewise']) and np.max(np.abs(colors['piecewise']))>0
        assert summaries['piecewise']['gate']
        rows.append(dict(name=name,kind=f['kind'],check=f['check'],variants=summaries,
                         forward_rgb_and_aux_identical=True))
    regression=json.loads((OUT/'prior_regression.json').read_text())
    assert regression['module_sha256']==receipt['module_sha256'] and regression['all_gates']
    assert sha(ROOT/'regress_tgs_piecewise.py')==regression['script_sha256']
    for row in regression['rows']:
        assert row['gate'] and row['forward_max_difference']==0
        with np.load(OUT/('regression_'+row['name']+'.npz')) as z:
            assert np.array_equal(z['native_gradient'],row['native_gradient'])
            assert np.array_equal(z['reference_gradient'],row['reference_gradient'])
    summary=dict(evidence_audit_passed=True,rows=rows,probe_step_pairs=pairs,prior_sealed_file_checks=old_files,
                 prior_rgb_regressions=len(regression['rows']),all_piecewise_gates=True,
                 geometry_fixed_failed_cases=[r['name'] for r in rows if not r['variants']['geometry_fixed']['gate']],
                 training_performed=False,quality_gain_established=False)
    (OUT/'summary_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['evidence_audit_passed','probe_step_pairs','prior_sealed_file_checks',
        'prior_rgb_regressions','all_piecewise_gates','geometry_fixed_failed_cases']}))


if __name__=='__main__':main()
