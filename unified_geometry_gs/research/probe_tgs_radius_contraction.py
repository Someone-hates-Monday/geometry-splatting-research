"""Execute the frozen old/stress VJP gates without dropping failed cases."""
import hashlib
import importlib.util
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch

from tgs_mixed_reference import ROOT,Native,reference as rgb_reference
from tgs_aux_reference import AuxNative,reference as aux_reference
from tgs_mixed_numpy import evaluate as rgb_numpy,complex_gradient as rgb_complex
from tgs_aux_numpy import evaluate as aux_numpy,complex_gradient as aux_complex

OUT=ROOT/'experiments/tgs_radius_contraction_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(path,digest):
    assert sha(path)==digest
    spec=importlib.util.spec_from_file_location(path.stem,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def check_group(g,ref):
    norm=np.linalg.norm(ref)
    error=np.linalg.norm(g-ref)/max(norm,1e-12)
    absolute=float(np.max(np.abs(g-ref))) if len(g) else 0.
    return dict(relative=float(error),absolute_max=absolute,reference_l2=float(norm),
                gate=bool(error<=1e-3 if norm>1e-10 else absolute<=1e-6))


def main():
    assert not (OUT/'results.json').exists() and not (OUT/'progress.json').exists(),'Version subsequent runs explicitly'
    plan=json.loads((OUT/'input_plan.json').read_text())
    receipt=json.loads((OUT/'build_receipt.json').read_text())
    assert plan['protocol_sha256']==receipt['protocol_sha256']==sha(ROOT/'TGS_RADIUS_CONTRACTION_PROTOCOL_V1.md')
    old_receipt=json.loads((ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/build_receipt.json').read_text())
    mods={'piecewise':load(Path(old_receipt['module_file']),old_receipt['module_sha256']),
          'contraction':load(Path(receipt['module_file']),receipt['module_sha256'])}
    execution=dict(input_plan_sha256=sha(OUT/'input_plan.json'),script_sha256=sha(Path(__file__)),
                   module_sha256={k:sha(Path(m.__file__)) for k,m in mods.items()},training=False)
    (OUT/'execution_plan.json').write_text(json.dumps(execution,indent=2)+'\n')
    records=[];started=time.time()
    cases=[('old',e) for e in plan['old_cases']]+[('stress',e) for e in plan['stress_cases']]
    for ci,(family,entry) in enumerate(cases):
        path=ROOT/entry['input'];assert sha(path)==entry['sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for key in ['n','w','h','degree']:f[key]=int(f[key])
        for key in ['name','kind','check']:f[key]=str(f[key])
        ref=aux_reference if f['kind']=='aux' else rgb_reference
        np_ref=aux_numpy if f['kind']=='aux' else rgb_numpy
        np_grad=aux_complex if f['kind']=='aux' else rgb_complex
        cls=AuxNative if f['kind']=='aux' else Native
        patterns=f['patterns'] if family=='stress' else np.array([f['pixel_grad']])
        references=[]
        for pattern in patterns:
            f['pixel_grad']=pattern
            x=torch.tensor(f['theta'].astype(float),requires_grad=True)
            loss,samples,meta=ref(x,f);g=torch.autograd.grad(loss,x)[0].numpy()
            cs=np_grad(f['theta'],f);nl,ns,fp=np_ref(f['theta'].astype(float),f)
            assert fp==meta['fingerprint'] and abs(float(loss.detach())-nl)<1e-9
            assert np.linalg.norm(g-cs)<=1e-8*max(np.linalg.norm(g),1)
            references.append((float(loss.detach()),samples.detach().numpy(),g,meta,cs))
        images={};aux_images={}
        for variant,mod in mods.items():
            f['pixel_grad']=patterns[0]
            native=cls(mod,f);data=native.forward(f['theta'])
            rows=[];gradients=[];fds=[]
            for pi,pattern in enumerate(patterns):
                f['pixel_grad']=pattern
                native.grad.zero_()
                for (x,y),weight in zip(f['pixels'],pattern):native.grad[:,y,x]=native.t(weight)
                loss,samples,exact,meta,cs=references[pi]
                g,arrays=native.backward(data);gradients.append(g)
                n=f['n'];groups=dict(geometry=check_group(g[:9*n],exact[:9*n]),
                                    sigma_opacity=check_group(g[9*n:11*n],exact[9*n:11*n]))
                if f['degree']>=0:groups['SH']=check_group(g[11*n:],exact[11*n:])
                if f['check']=='homothety_nullspace':
                    directional=float(g@f['special_direction']);gate=abs(directional)<=1e-6
                else:directional=None;gate=all(v['gate'] for v in groups.values())
                forward_error=float(np.max(np.abs(data[-1]-samples)))
                gate=gate and forward_error<=3e-5 and bool((data[1][3]>0).all())
                rows.append(dict(pattern=pi,groups=groups,direction=directional,forward_max_abs=forward_error,
                                 reference_loss=loss,native_loss=float((data[-1].astype(float)*(f['aux_grad'] if f['kind']=='aux' else pattern)).sum()),
                                 native_gradient=g.tolist(),reference_gradient=exact.tolist(),complex_step_gradient=cs.tolist(),
                                 metadata=meta,gate=bool(gate)))
                if family=='stress':
                    for h in f['fd_steps']:
                        for di,d in enumerate(f['fd_directions'][pi]):
                            plus=(f['theta'].astype(float)+h*d).astype(np.float32)
                            minus=(f['theta'].astype(float)-h*d).astype(np.float32)
                            lp,rp,_,_,_,sp=native.forward(plus);lm,rm,_,_,_,sm=native.forward(minus)
                            cp,_,fpp=np_ref(plus.astype(float),f);cm,_,fpm=np_ref(minus.astype(float),f)
                            actual=(plus.astype(float)-minus.astype(float))/(2*h)
                            smooth=fpp==fpm==meta['fingerprint'] and bool((rp[3]>0).all()) and bool((rm[3]>0).all())
                            fds.append(dict(pattern=pi,step=float(h),direction=di,actual_direction=actual.tolist(),
                                expected=float(exact@actual),native_fd=(lp-lm)/(2*h),cpu_fd=float(cp-cm)/(2*h),
                                native_plus_loss=lp,native_minus_loss=lm,cpu_plus_loss=float(cp),cpu_minus_loss=float(cm),
                                plus_samples=sp.tolist(),minus_samples=sm.tolist(),plus_fp=fpp,minus_fp=fpm,smooth=bool(smooth)))
            images[variant]=data[1][1].cpu().numpy();aux_images[variant]=data[1][2].cpu().numpy()
            old_equal=None
            if family=='old':
                p=ROOT/entry['old_output'];assert sha(p)==entry['old_output_sha256']
                with np.load(p) as z:old_equal=bool(np.array_equal(images[variant],z['native_rgb']))
            all_smooth=all(i['smooth'] for i in fds)
            row=dict(name=f['name'],family=family,variant=variant,patterns=rows,finite_differences=fds,
                     whole_rgb_matches_prior=old_equal,all_fd_branches_preserved=all_smooth,
                     gate=bool(all(r['gate'] for r in rows) and all_smooth and old_equal is not False))
            records.append(row)
            np.savez_compressed(OUT/(variant+'_'+f['name']+'.npz'),native_rgb=images[variant],native_aux=aux_images[variant],
                native_samples=data[-1],native_gradients=np.array(gradients),reference_gradients=np.array([v[2] for v in references]))
            (OUT/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        identical=bool(np.array_equal(images['piecewise'],images['contraction']) and np.array_equal(aux_images['piecewise'],aux_images['contraction']))
        for r in records[-2:]:r['whole_rgb_aux_identical_between_variants']=identical;r['gate']=r['gate'] and identical
        if ci<18 or ci%6==5:
            print(json.dumps(dict(case=ci+1,total=len(cases),name=f['name'],
                piecewise_gate=records[-2]['gate'],contraction_gate=records[-1]['gate'],
                contraction_max_geometry_relative=max(p['groups']['geometry']['relative'] for p in records[-1]['patterns']))),flush=True)
    final=dict(input_plan_sha256=sha(OUT/'input_plan.json'),execution_plan_sha256=sha(OUT/'execution_plan.json'),records=records,
        old_cases=18,stress_cases=48,stress_vjps_per_case=8,
        all_contraction_precision_gates=all(r['gate'] for r in records if r['variant']=='contraction'),
        contraction_failed_cases=[r['name'] for r in records if r['variant']=='contraction' and not r['gate']],
        piecewise_failed_cases=[r['name'] for r in records if r['variant']=='piecewise' and not r['gate']],
        native_fd_probe_step_pairs=sum(len(r['finite_differences']) for r in records),
        seconds=time.time()-started,training=False)
    final['performance_measurement_allowed']=final['all_contraction_precision_gates']
    (OUT/'results.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({k:final[k] for k in ['all_contraction_precision_gates','contraction_failed_cases','piecewise_failed_cases',
                                         'native_fd_probe_step_pairs','seconds','performance_measurement_allowed']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'execution_failure.txt').write_text(traceback.format_exc())
        raise
