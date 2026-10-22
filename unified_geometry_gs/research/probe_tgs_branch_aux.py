"""Preserved branch counterexamples and smooth auxiliary-map gradient probes.

No optimizer. Tie geometry is judged by a zero directional-derivative invariant,
not a fabricated unique gradient at a nonsmooth point.
"""
import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from tgs_mixed_reference import ROOT, Native, load_module, reference as rgb_reference
from tgs_aux_reference import AuxNative,reference as aux_reference
from probe_tgs_mixed_backward import relative

OUT=ROOT/'experiments/tgs_branch_aux_v1'


def single(name,pixels,points_scale=1.):
    w=h=256
    screen=np.array([[32.,32.],[192.,32.],[32.,152.]])
    p=np.c_[(((2*screen+1)/256-1)*2)*points_scale,np.full(3,2.)].astype(np.float32)
    projection=np.array([[1/points_scale,0,0,0],[0,1/points_scale,0,0],[0,0,1,0],[0,0,1,0]],dtype=np.float32)
    return dict(name=name,n=1,w=w,h=h,degree=-1,theta=np.r_[p.ravel(),1.3,.65].astype(np.float32),
                view=np.eye(4,dtype=np.float32),projection=projection,campos=np.zeros(3,dtype=np.float32),
                tx=points_scale,ty=points_scale,colors=np.array([[.7,.4,.2]],dtype=np.float32),
                background=np.array([.08,.13,.21],dtype=np.float32),pixels=np.array(pixels,dtype=int),
                pixel_grad=np.tile(np.array([1.,-.3,.5],dtype=np.float32),(len(pixels),1)),
                kind='rgb',check='smooth',steps=np.array([.003,.001,.0003]))


def fixtures():
    cap=single('alpha_saturated',[[76,72],[79,70]])
    cap['theta'][-2:]=[.02,.9999];cap['check']='saturated_zero'
    cap['steps']=np.array([3e-5,1e-5,3e-6])
    yield cap
    tie=single('two_edge_tie',[[60,60]])
    tie['check']='homothety_nullspace'
    # Invert the same real-valued pinhole denominator used by the reference.
    pivot=((2*np.array([60,60])+1)/256-1)*(2+1e-7)
    d=np.zeros(len(tie['theta']));d[:9].reshape(3,3)[:,:2]=tie['theta'][:9].reshape(3,3)[:,:2]-pivot
    tie['special_direction']=d
    yield tie
    early=single('early_stop_skips_fourth',[[76,72],[79,70]])
    base=early['theta'][:9].reshape(3,3)
    zs=np.array([2.,2.4,2.8,3.2,3.6])
    points=np.array([base*z/2 for z in zs],dtype=np.float32)
    early.update(n=5,theta=np.r_[points.ravel(),np.full(5,.02),np.full(5,.95)].astype(np.float32),
                 colors=np.array([[.7,.4,.2],[.2,.7,.3],[.4,.2,.8],[.5,.3,.6],[.9,.1,.4]],dtype=np.float32))
    yield early
    # Reuse previously frozen three-layer geometry, not its gradient outcomes.
    with np.load(ROOT/'experiments/tgs_mixed_backward_v1/three_layer_color_input.npz') as z:
        base={k:z[k].copy() for k in z.files if not k.startswith('probe_')}
    for key in ['n','w','h','degree']:base[key]=int(base[key])
    weights=np.array([1.,.7,-.4,1.1],dtype=np.float32)
    for name,cols in [('alpha',[1]),('depth',[0]),('normal',[2,3,4]),('median',[5]),('distortion',[6])]:
        f=dict(base,name='aux_'+name,kind='aux',check='smooth',pixel_grad=np.zeros((4,3),dtype=np.float32),
               steps=np.array([.01,.003,.001]))
        g=np.zeros((4,7),dtype=np.float32)
        g[:,cols]=weights[:,None]*(np.array([.6,-.3,.8]) if name=='normal' else 1.)
        f['aux_grad']=g
        yield f
    # Same resolvable screen footprint but world cross norm 5e-5 < 1e-4.
    scale=np.sqrt(5e-5/4.6875)
    clipped=single('aux_normal_floor',[[72,65],[135,55],[55,105],[104,91]],scale)
    clipped.update(kind='aux',pixel_grad=np.zeros((4,3),dtype=np.float32),steps=np.array([3e-6,1e-6,3e-7]))
    g=np.zeros((4,7),dtype=np.float32);g[:,4]=weights
    clipped['aux_grad']=g
    yield clipped


def freeze():
    assert not OUT.exists(),'Version later attempts explicitly; do not overwrite'
    OUT.mkdir(parents=True)
    plan=dict(scope='saturation, tie-nullspace, early stop and auxiliary maps; no training',scenes=[],
              interpretation='smooth branches get VJP comparisons; ties only a proven invariant; no intent inferred about surrogate gradients',
              gates=dict(group_rel=1e-3,zero_abs=1e-6,forward_abs=3e-5),
              script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for f in fixtures():
        ref=aux_reference if f['kind']=='aux' else rgb_reference
        _,_,meta=ref(torch.tensor(f['theta'].astype(float)),f)
        path=OUT/(f['name']+'_input.npz');np.savez_compressed(path,**f)
        plan['scenes'].append(dict(name=f['name'],input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metadata=meta))
    (OUT/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(stage='frozen_before_cuda',scenes=len(plan['scenes']))),flush=True)
    return plan


def run(mod,variant,plan):
    rows=[]
    for scene in plan['scenes']:
        with np.load(OUT/(scene['name']+'_input.npz')) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        ref=aux_reference if f['kind']=='aux' else rgb_reference
        cls=AuxNative if f['kind']=='aux' else Native
        x=torch.tensor(f['theta'].astype(float),requires_grad=True)
        loss,samples,meta=ref(x,f);exact=torch.autograd.grad(loss,x)[0].numpy()
        native=cls(mod,f);data=native.forward(f['theta']);g,arrays=native.backward(data)
        if f['check']=='homothety_nullspace':
            ds=np.array([f['special_direction']]);labels=['homothety']
        else:
            ds=np.eye(len(x));labels=[f'coordinate_{j}' for j in range(len(x))]
        fd=[]
        for h in f['steps']:
            items=[]
            for d,label in zip(ds,labels):
                plus=(f['theta'].astype(float)+h*d).astype(np.float32)
                minus=(f['theta'].astype(float)-h*d).astype(np.float32)
                lp,rp,_,_,_,sp=native.forward(plus);lm,rm,_,_,_,sm=native.forward(minus)
                cp,_,mp=ref(torch.tensor(plus.astype(float)),f);cm,_,mm=ref(torch.tensor(minus.astype(float)),f)
                smooth=mp['fingerprint']==mm['fingerprint']==meta['fingerprint'] and bool((rp[3]>0).all()) and bool((rm[3]>0).all())
                items.append(dict(probe=label,expected_linearized=float(exact@((plus.astype(float)-minus.astype(float))/(2*h))),
                                  native_fd=(lp-lm)/(2*h),cpu_fd=float(cp-cm)/(2*h),
                                  native_plus_loss=lp,native_minus_loss=lm,cpu_plus_loss=float(cp),cpu_minus_loss=float(cm),
                                  plus_samples=sp.tolist(),minus_samples=sm.tolist(),plus_meta=mp,minus_meta=mm,smooth=bool(smooth)))
            fd.append(dict(step=float(h),items=items))
        n=f['n'];geometry=relative(g[:9*n],exact[:9*n]);scalar=relative(g[9*n:],exact[9*n:])
        forward_error=float(np.max(np.abs(data[-1]-samples.detach().numpy())))
        smooth=all(i['smooth'] for s in fd for i in s['items'])
        if f['check']=='homothety_nullspace':
            native_direction=float(g@f['special_direction']);true_direction=float(exact@f['special_direction'])
            gate=abs(native_direction)<1e-6
        else:
            native_direction=true_direction=None
            geom_ok=geometry<1e-3 if np.linalg.norm(exact[:9*n])>1e-10 else np.max(np.abs(g[:9*n]))<1e-6
            scalar_ok=scalar<1e-3 if np.linalg.norm(exact[9*n:])>1e-10 else np.max(np.abs(g[9*n:]))<1e-6
            gate=geom_ok and scalar_ok and smooth and forward_error<3e-5
        row=dict(name=f['name'],variant=variant,kind=f['kind'],check=f['check'],forward_max_abs=forward_error,
                 geometry_relative_error=geometry,scalar_relative_error=scalar,native_gradient=g.tolist(),exact_gradient=exact.tolist(),
                 native_gradient_max_abs=float(np.max(np.abs(g))),exact_gradient_max_abs=float(np.max(np.abs(exact))),
                 native_direction=native_direction,reference_direction=true_direction,metadata=meta,
                 all_perturbations_smooth=smooth,gate=bool(gate),finite_differences=fd,native_loss=data[0],cpu_loss=float(loss.detach()))
        rows.append(row)
        np.savez_compressed(OUT/(variant+'_'+f['name']+'.npz'),native_samples=data[-1],reference_samples=samples.detach().numpy(),
            native_rgb=data[1][1].cpu().numpy(),native_aux=data[1][2].cpu().numpy(),**{f'native_grad_{j}':a for j,a in enumerate(arrays)})
        (OUT/(variant+'_progress.json')).write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps({k:row[k] for k in ['variant','name','forward_max_abs','geometry_relative_error','scalar_relative_error',
                         'native_direction','reference_direction','all_perturbations_smooth','gate']}),flush=True)
    result=dict(variant=variant,rows=rows,module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
                all_gates=all(r['gate'] for r in rows),training_performed=False)
    (OUT/(variant+'_results.json')).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--piecewise',action='store_true');args=parser.parse_args()
    try:
        if args.piecewise:
            plan=json.loads((OUT/'protocol.json').read_text())
            from tgs_piecewise_backend import load_piecewise
            run(load_piecewise(),'piecewise',plan)
        else:
            plan=freeze();run(load_module('fixed'),'geometry_fixed',plan)
    except BaseException:
        if OUT.exists():(OUT/'execution_failure.txt').write_text(traceback.format_exc())
        raise
