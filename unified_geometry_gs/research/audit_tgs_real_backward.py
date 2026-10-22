"""Independent real-subset VJP/crop audit; optional exact checkpoint tensor check."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tgs_mixed_numpy import evaluate,complex_gradient

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_real_backward_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def checkpoint_check():
    import torch
    from audit_tgs_real_checkpoint import load_input
    model,_,_,_,_=load_input()
    model={k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in model.items()}
    plan=json.loads((OUT/'protocol.json').read_text())
    for s in plan['scenes']:
        with np.load(OUT/(s['name']+'_input.npz')) as z:
            ids=z['parent_ids']
            expected=np.r_[model['triangles_points'][ids].numpy().ravel(),
                (.01+torch.exp(model['sigma'][ids])).numpy().ravel(),
                torch.sigmoid(model['opacity'][ids]).numpy().ravel(),
                torch.cat([model['features_dc'][ids],model['features_rest'][ids]],1).numpy().ravel()]
            assert np.array_equal(z['theta'],expected)
    result=dict(passed=True,scope='all points, activated sigma/opacity and all degree3 SH coefficients of both saved subsets',
                all_saved_subset_tensors_bitwise_equal_to_checkpoint=True,training=False)
    (OUT/'checkpoint_tensor_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


def main():
    plan=json.loads((OUT/'protocol.json').read_text())
    results=json.loads((OUT/'results.json').read_text())
    assert sha(OUT/'protocol.json')==results['protocol_sha256']
    assert sha(ROOT/'probe_tgs_real_backward.py')==plan['script_sha256']==results['script_sha256']
    assert sha(ROOT/'tgs_mixed_reference.py')==plan['reference_sha256']
    selection=json.loads((ROOT/'sources/tgs_real_checkpoint_v1/selection.json').read_text())
    checkpoint=ROOT/'sources/tgs_real_checkpoint_v1'/selection['scene']/'point_cloud'/selection['iteration']/'point_cloud_state_dict.pt'
    assert sha(checkpoint)==plan['checkpoint_sha256']
    input_audit=json.loads((ROOT/'experiments/tgs_real_event_v1/input_audit.json').read_text())
    summaries=[]
    for scene in plan['scenes']:
        path=OUT/(scene['name']+'_input.npz');assert sha(path)==scene['input_sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        camera=scene['camera'];assert camera['split']=='train'
        ci=next(i for i,c in enumerate(input_audit['selected_cameras']) if c['img_name']==camera['img_name'])
        assert camera==input_audit['selected_cameras'][ci]
        with np.load(ROOT/f'experiments/tgs_real_event_v1/baseline_{ci}.npz') as z:
            assert np.array_equal(f['view'],z['view'].T)
            assert np.array_equal(f['campos'],z['camera_center'])
            assert np.array_equal(f['full_projection'],z['projection'].T.astype(float))
        assert f['parent_ids'].tolist()==scene['ids'] and len(set(scene['ids']))==4
        assert np.array_equal(f['pixels'],f['global_pixels']-f['crop_origin'])
        crop=f['full_projection'].copy();w,h=f['original_resolution'];x0,y0=f['crop_origin']
        for dim,(size,start) in enumerate([(w,x0),(h,y0)]):
            a=size/128;b=(size-2*start)/128-1
            crop[dim]=a*f['full_projection'][dim]+b*f['full_projection'][3]
            crop[dim,3]+=b*1e-7
        assert np.array_equal(crop.astype(np.float32),f['projection'])
        n=int(f['n']);p=f['theta'][:9*n].astype(float).reshape(n,3,3)
        homogeneous=np.concatenate([p,np.ones((n,3,1))],-1)
        full=homogeneous@f['full_projection'].T;local=homogeneous@f['projection'].astype(float).T
        fullxy=(full[...,:2]/(full[...,3:]+1e-7)+1)*np.array([w/2,h/2])-.5
        localxy=(local[...,:2]/(local[...,3:]+1e-7)+1)*64-.5
        crop_rounding=float(np.max(np.abs(localxy-(fullxy-[x0,y0]))))
        assert crop_rounding<1e-3
        cpu_loss,rgb,fp=evaluate(f['theta'].astype(float),f)
        cs=complex_gradient(f['theta'],f)
        variants={r['variant']:r for r in results['records'] if r['name']==scene['name']}
        images={};errors={};fd_gate={}
        for variant,row in variants.items():
            exact=np.array(row['exact_gradient']);g=np.array(row['native_gradient'])
            assert rel(cs,exact)<1e-10 and abs(cpu_loss-row['cpu_loss'])<2e-12
            assert fp==row['metadata']['fingerprint']==scene['metadata']['fingerprint']
            with np.load(OUT/(variant+'_'+scene['name']+'.npz')) as z:
                assert np.array_equal(z['native_grad_0'].ravel(),g[:9*n])
                assert np.array_equal(z['native_grad_1'].ravel(),g[9*n:10*n])
                assert np.array_equal(z['native_grad_3'].ravel(),g[10*n:11*n])
                assert np.array_equal(z['native_grad_4'].ravel(),g[11*n:])
                assert np.max(np.abs(rgb-z['cpu_samples']))<2e-12
                forward_error=float(np.max(np.abs(z['native_samples']-rgb)))
                assert abs(forward_error-row['forward_max_abs'])<1e-12
                images[variant]=z['native_rgb'].copy()
            group=dict(geometry=rel(g[:9*n],cs[:9*n]),sigma_opacity=rel(g[9*n:11*n],cs[9*n:11*n]),SH=rel(g[11*n:],cs[11*n:]))
            for key,value in group.items():assert abs(value-row['gradients_relative'][key])<1e-10
            for step in row['finite_differences']:
                expecteds=[];native_fds=[];cpu_fds=[]
                for j,item in enumerate(step['rows']):
                    hh=step['step'];d=f['probe_directions'][j]
                    plus=(f['theta'].astype(float)+hh*d).astype(np.float32)
                    minus=(f['theta'].astype(float)-hh*d).astype(np.float32)
                    lp,_,fpp=evaluate(plus.astype(float),f);lm,_,fpm=evaluate(minus.astype(float),f)
                    assert fpp==fpm==fp and item['smooth']
                    assert abs(lp-item['cpu_plus_loss'])<3e-12 and abs(lm-item['cpu_minus_loss'])<3e-12
                    expected=float(cs@((plus.astype(float)-minus.astype(float))/(2*hh)))
                    assert abs(expected-item['expected'])<1e-9
                    native_fd=(item['native_plus_loss']-item['native_minus_loss'])/(2*hh)
                    cpu_fd=(lp-lm)/(2*hh)
                    assert abs(native_fd-item['native_fd'])<1e-12 and abs(cpu_fd-item['cpu_fd'])<2e-7
                    expecteds.append(expected);native_fds.append(native_fd);cpu_fds.append(cpu_fd)
                    for side in ['plus','minus']:
                        observed=np.array(item[side+'_samples'])
                        assert abs((observed*f['pixel_grad'].astype(float)).sum()-item['native_'+side+'_loss'])<2e-13
                assert abs(rel(np.array(native_fds),np.array(expecteds))-step['native_fd_relative'])<1e-9
                assert abs(rel(np.array(cpu_fds),np.array(expecteds))-step['cpu_fd_relative'])<1e-8
            middle=row['finite_differences'][1]
            fd_gate[variant]=middle['native_fd_relative']
            gate=all(e<1e-3 for e in group.values()) and row['forward_max_abs']<3e-5 \
                and row['all_perturbations_smooth'] and middle['native_fd_relative']<1e-2 and middle['cpu_fd_relative']<1e-3
            assert bool(gate)==row['gate']
            if variant=='fixed':assert gate
            errors[variant]=group
        assert np.array_equal(images['control'],images['fixed'])
        summaries.append(dict(name=scene['name'],ids=scene['ids'],geometry_and_appearance_errors=errors,
                              complex_step_vs_autograd=rel(cs,np.array(variants['fixed']['exact_gradient'])),
                              crop_projection_rounding_max_pixels=crop_rounding,native_fd_at_middle=fd_gate,
                              steps=scene['steps'],forward_images_identical=True))
    summary=dict(passed=True,scenes=summaries,total_probe_step_pairs=results['total_probe_step_pairs'],
                 checkpoint_sha256=plan['checkpoint_sha256'],all_fixed_gates=True,
                 real_data_scope='two geometry-selected four-triangle subsets and three pixels per crop; not a full scene or quality test',
                 training_performed=False,quality_gain_established=False)
    (OUT/'summary_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-checkpoint',action='store_true')
    args=parser.parse_args()
    if args.check_checkpoint:checkpoint_check()
    else:main()
