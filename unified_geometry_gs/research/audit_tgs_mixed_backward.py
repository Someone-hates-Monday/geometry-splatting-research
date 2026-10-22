"""Read-only evidence audit plus independent complex-step differentiation."""
import hashlib
import json
from pathlib import Path

import numpy as np
from tgs_mixed_numpy import evaluate,complex_gradient

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_mixed_backward_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def main():
    # Ensure that read-only native imports did not rewrite any prior evidence.
    old=json.loads((ROOT/'experiments/tgs_backward_v1/evidence_manifest.json').read_text())
    for e in old['files']:
        assert sha(ROOT/e['path'])==e['sha256']
    plan=json.loads((OUT/'protocol.json').read_text())
    results=json.loads((OUT/'results.json').read_text())
    assert sha(OUT/'protocol.json')==results['protocol_sha256']
    assert sha(ROOT/'probe_tgs_mixed_backward.py')==plan['script_sha256']==results['script_sha256']
    assert sha(ROOT/'tgs_mixed_reference.py')==plan['reference_sha256']==results['reference_sha256']
    for path,digest in plan['modules'].values():assert sha(ROOT/path)==digest
    rows=[]
    for scene in plan['scenes']:
        name=scene['name'];path=OUT/(name+'_input.npz')
        assert sha(path)==scene['input_sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        loss,rgb,fp=evaluate(f['theta'].astype(float),f)
        cs=complex_gradient(f['theta'],f)
        variants={r['variant']:r for r in results['records'] if r['name']==name}
        images={}
        for variant,row in variants.items():
            exact=np.array(row['exact_gradient'])
            g=np.array(row['native_gradient']);n=int(f['n'])
            assert rel(cs,exact)<1e-11
            assert abs(loss-row['cpu_loss'])<1e-12 and fp==row['metadata']['fingerprint']
            with np.load(OUT/(variant+'_'+name+'.npz')) as z:
                assert np.max(np.abs(z['cpu_samples']-rgb))<1e-12
                samples=z['native_samples']
                assert abs((samples.astype(float)*f['pixel_grad'].astype(float)).sum()-row['native_loss'])<1e-14
                packed=np.r_[z['native_grad_0'].ravel(),z['native_grad_1'].ravel(),z['native_grad_3'].ravel(),
                             z['native_grad_4'].ravel() if int(f['degree'])>=0 else []]
                assert np.array_equal(packed,g)
                forward_error=float(np.max(np.abs(samples-rgb)))
                assert abs(forward_error-row['forward_max_abs'])<1e-12
                images[variant]=z['native_rgb'].copy()
            groups=dict(geometry=rel(g[:9*n],cs[:9*n]),sigma_opacity=rel(g[9*n:11*n],cs[9*n:11*n]))
            if int(f['degree'])>=0:groups['SH']=rel(g[11*n:],cs[11*n:])
            for key,value in groups.items():assert abs(value-row['gradients_relative'][key])<1e-11
            for step in row['finite_differences']:
                expected_values=[];native_fds=[];cpu_fds=[]
                for idx,item in enumerate(step['rows']):
                    h=step['step'];d=f['probe_directions'][idx]
                    plus=(f['theta'].astype(float)+h*d).astype(np.float32)
                    minus=(f['theta'].astype(float)-h*d).astype(np.float32)
                    lp,_,fpp=evaluate(plus.astype(float),f)
                    lm,_,fpm=evaluate(minus.astype(float),f)
                    assert fpp==fpm==fp and item['smooth']
                    assert abs(lp-item['cpu_plus_loss'])<2e-12
                    assert abs(lm-item['cpu_minus_loss'])<2e-12
                    expected=float(cs@((plus.astype(float)-minus.astype(float))/(2*h)))
                    assert abs(expected-item['expected'])<1e-10
                    assert abs((lp-lm)/(2*h)-item['cpu_fd'])<2e-9
                    native_fd=(item['native_plus_loss']-item['native_minus_loss'])/(2*h)
                    assert abs(native_fd-item['native_fd'])<1e-14
                    expected_values.append(expected);native_fds.append(native_fd);cpu_fds.append((lp-lm)/(2*h))
                    for side in ['plus','minus']:
                        observed=np.array(item[side+'_samples'])
                        native_loss=(observed*f['pixel_grad'].astype(float)).sum()
                        assert abs(native_loss-item['native_'+side+'_loss'])<2e-13
                assert abs(rel(np.array(native_fds),np.array(expected_values))-step['native_fd_relative'])<1e-10
                assert abs(rel(np.array(cpu_fds),np.array(expected_values))-step['cpu_fd_relative'])<1e-9
            middle=next(s for s in row['finite_differences'] if s['step']==.003)
            gate=all(v<1e-3 for v in groups.values()) and forward_error<3e-5 and row['all_perturbations_smooth'] \
                and middle['native_fd_relative']<1e-3 and middle['cpu_fd_relative']<1e-4
            assert bool(gate)==row['gate']
            if variant=='fixed':assert gate
        assert np.array_equal(images['control'],images['fixed'])
        rows.append(dict(name=name,complex_step_vs_autograd_relative=rel(cs,np.array(variants['fixed']['exact_gradient'])),
                         fixed_gradients_relative=variants['fixed']['gradients_relative'],
                         native_fd_at_gate=next(s['native_fd_relative'] for s in variants['fixed']['finite_differences'] if s['step']==.003),
                         forward_images_identical=True))
    summary=dict(passed=True,scenes=rows,total_probe_step_pairs=results['total_probe_step_pairs'],
                 previous_sealed_files_unchanged=len(old['files']),
                 all_fixed_gates=True,training_performed=False,quality_gain_established=False)
    (OUT/'summary_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
