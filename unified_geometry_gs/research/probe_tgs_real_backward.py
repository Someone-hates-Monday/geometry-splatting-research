"""Bounded RGB VJP diagnosis on frozen subsets of a published real checkpoint.

Selects overlapping triangles using geometry and smooth-branch margins only,
before CUDA/VJP outcomes. Uses the two previously fixed training cameras and
128x128 crops with preserved pixel scale. No optimizer, target fitting or GT use.
"""
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from audit_tgs_real_checkpoint import load_input
from tgs_mixed_reference import ROOT, Native, load_module, reference
from probe_tgs_mixed_backward import directions, relative

OUT=ROOT/'experiments/tgs_real_backward_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare():
    assert not OUT.exists(),'Preserve previous real-subset diagnoses; explicitly version later runs'
    OUT.mkdir(parents=True)
    model,_,_,_,selection=load_input()
    model={k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in model.items()}
    input_audit=json.loads((ROOT/'experiments/tgs_real_event_v1/input_audit.json').read_text())
    sigma=(.01+torch.exp(model['sigma'])).numpy().ravel()
    opacity=torch.sigmoid(model['opacity']).numpy().ravel()
    all_points=model['triangles_points'].numpy()
    all_sh=torch.cat([model['features_dc'],model['features_rest']],1).numpy()
    assert all_sh.shape[1:]==(16,3)
    w,h=input_audit['actual_resolution']
    plan=dict(scope='fixed 4-triangle subsets of published stump iteration7000, two train-camera crops; no training',
              checkpoint_sha256=sha(ROOT/'sources/tgs_real_checkpoint_v1'/selection['scene']/'point_cloud'/selection['iteration']/'point_cloud_state_dict.pt'),
              script_sha256=sha(Path(__file__)),reference_sha256=sha(ROOT/'tgs_mixed_reference.py'),
              scene_selection='first SHA256-ranked eligible anchor with 4 overlapping, depth-separated, branch-safe triangles',
              depth_separation_min=.02,crop_size=128,base_steps=[.0003,.0001,.00003],
              step_reduction='halve all steps until largest-step probes preserve CPU branch; at most 8 halvings; before CUDA',
              gates=dict(group_relative=1e-3,forward_abs=3e-5,native_fd_at_middle_step=1e-2,cpu_fd_at_middle_step=1e-3,
                         every_probe_smooth=True),scenes=[],attempts=[])
    for ci in [2,3]:
        cam=input_audit['selected_cameras'][ci]
        assert cam['split']=='train'
        with np.load(ROOT/f'experiments/tgs_real_event_v1/baseline_{ci}.npz') as z:
            radii=z['radii'];native_radius=z['inradius'];view=z['view'].T.astype(float)
            projection=z['projection'].T.astype(float);campos=z['camera_center'].copy()
        eligible=np.flatnonzero((radii>0)&(native_radius>=4)&(opacity>=.05))
        points=all_points[eligible].astype(float)
        hom=np.concatenate([points,np.ones((len(points),3,1))],-1)@projection.T
        xy=(hom[...,:2]/(hom[...,3:]+1e-7)+1)*[w/2,h/2]-.5
        edge=np.roll(xy,-1,1)-xy
        length=np.linalg.norm(edge,axis=-1)
        perimeter=length.sum(1)
        inc=(np.roll(length,-1,1)[...,None]*xy).sum(1)/perimeter[:,None]
        D=(xy[:,1,0]-xy[:,0,0])*(xy[:,2,1]-xy[:,0,1])-(xy[:,1,1]-xy[:,0,1])*(xy[:,2,0]-xy[:,0,0])
        rr=np.abs(D)/perimeter
        normal=np.stack([edge[...,1],-edge[...,0]],-1)/length[...,None]*np.sign(D)[:,None,None]
        offset=-(normal*xy).sum(-1)
        zmean=(np.c_[points.mean(1),np.ones(len(points))]@view.T)[:,2]
        positive=(hom[...,3]>.2).all(1)
        hashes=[hashlib.sha256(('tgs_real_vjp_v1/'+cam['img_name']+'/'+str(int(i))).encode()).hexdigest() for i in eligible]
        ordered=sorted(range(len(eligible)),key=lambda j:hashes[j])
        anchors=[j for j in ordered if positive[j] and 8<=rr[j]<=24
                 and 8<inc[j,0]<w-8 and 8<inc[j,1]<h-8][:64]
        chosen=None
        for anchor in anchors:
            for shift in [[3.1,-1.8],[-3.2,2.3],[2.2,3.7]]:
                pixels=np.rint(inc[anchor]+shift).astype(int)+np.array([[0,0],[1,0],[0,1]])
                if np.any(pixels<0) or np.any(pixels[:,0]>=w) or np.any(pixels[:,1]>=h):continue
                phi=np.einsum('sd,nkd->nsk',pixels,normal)+offset[:,None]
                sorted_phi=np.sort(phi,axis=-1)
                maxphi=sorted_phi[...,-1]
                u=-maxphi/rr[:,None]
                safe_u=np.maximum(u,1e-30)
                a=opacity[eligible,None]*safe_u**sigma[eligible,None]
                mask=positive&(maxphi<-1.5).all(1)&((sorted_phi[...,-1]-sorted_phi[...,-2])>.75).all(1) \
                    &(a>.03).all(1)&(a<.9).all(1)
                if not mask[anchor]:continue
                picked=[anchor]
                for j in ordered:
                    if j==anchor or not mask[j]:continue
                    if min(abs(zmean[j]-zmean[k]) for k in picked)>.02:
                        picked.append(j)
                    if len(picked)==4:break
                plan['attempts'].append(dict(camera=cam['img_name'],anchor=int(eligible[anchor]),shift=shift,
                                              eligible_overlap=int(mask.sum()),selected=len(picked)))
                if len(picked)<4:continue
                ids=eligible[picked]
                x0=int(np.clip(pixels[0,0]-64,0,w-128));y0=int(np.clip(pixels[0,1]-64,0,h-128))
                crop=projection.copy()
                for dim,(full,start) in enumerate([(w,x0),(h,y0)]):
                    scale=full/128;b=(full-2*start)/128-1
                    crop[dim]=scale*projection[dim]+b*projection[3]
                    crop[dim,3]+=b*1e-7
                theta=np.r_[all_points[ids].ravel(),sigma[ids],opacity[ids],all_sh[ids].ravel()].astype(np.float32)
                f=dict(name='real_'+cam['img_name'],n=4,w=128,h=128,degree=3,theta=theta,
                       pixels=pixels-[x0,y0],pixel_grad=(np.array([1.,.7,-.4])[:,None]*[1.,-.3,.5]).astype(np.float32),
                       view=view.astype(np.float32),projection=crop.astype(np.float32),campos=campos.astype(np.float32),
                       colors=np.ones((4,3),dtype=np.float32),background=np.zeros(3,dtype=np.float32),
                       tx=float(cam['width']/(2*cam['fx'])*128/w),ty=float(cam['height']/(2*cam['fy'])*128/h))
                _,_,meta=reference(torch.tensor(theta.astype(float)),f)
                if not np.all(meta['fingerprint']['included']) or np.any(meta['fingerprint']['capped']):continue
                if np.min(np.abs(meta['raw_colors']))<.02 or np.min(meta['transmittance'])<5e-4:continue
                ds=directions(f);factor=None
                for halves in range(9):
                    step=.0003*2**(-halves);stable=True
                    for _,d in ds:
                        for sign in [-1,1]:
                            x=(theta.astype(float)+sign*step*d).astype(np.float32)
                            _,_,m=reference(torch.tensor(x.astype(float)),f)
                            if m['fingerprint']!=meta['fingerprint'] or m['edge_gap_min']<.3:
                                stable=False;break
                        if not stable:break
                    if stable:factor=2**(-halves);break
                if factor is None:continue
                chosen=(f,ids,pixels,[x0,y0],factor,meta,ds)
                break
            if chosen is not None:break
        if chosen is None:
            (OUT/'selection_failure.json').write_text(json.dumps(plan,indent=2)+'\n')
            raise RuntimeError('No safe four-layer subset found within frozen anchor/shift budget')
        f,ids,pixels,origin,factor,meta,ds=chosen
        path=OUT/(f['name']+'_input.npz')
        np.savez_compressed(path,**f,parent_ids=ids,global_pixels=pixels,crop_origin=np.array(origin),
                            full_projection=projection,original_resolution=np.array([w,h]),
                            probe_directions=np.array([d for _,d in ds]),probe_names=np.array([s for s,_ in ds]))
        plan['scenes'].append(dict(name=f['name'],camera=cam,ids=ids.tolist(),input_sha256=sha(path),
                                  steps=[s*factor for s in plan['base_steps']],step_factor=factor,metadata=meta))
    (OUT/'protocol.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(stage='real_inputs_frozen_before_cuda',scenes=len(plan['scenes']),
                          triangles_per_scene=4,protocol_sha256=sha(OUT/'protocol.json'))),flush=True)
    return plan


def main():
    start=time.time();plan=prepare();records=[]
    for variant in ['control','fixed']:
        mod=load_module(variant)
        for scene in plan['scenes']:
            path=OUT/(scene['name']+'_input.npz');assert sha(path)==scene['input_sha256']
            with np.load(path) as z:f={k:z[k].copy() for k in z.files}
            for key in ['n','w','h','degree']:f[key]=int(f[key])
            x=torch.tensor(f['theta'].astype(float),requires_grad=True)
            loss,rgb,meta=reference(x,f);exact=torch.autograd.grad(loss,x)[0].numpy()
            native=Native(mod,f);data=native.forward(f['theta']);gpu,arrays=native.backward(data)
            diffs=[]
            for h in scene['steps']:
                rows=[]
                for label,d in zip(f['probe_names'],f['probe_directions']):
                    plus=(f['theta'].astype(float)+h*d).astype(np.float32)
                    minus=(f['theta'].astype(float)-h*d).astype(np.float32)
                    expected=float(exact@((plus.astype(float)-minus.astype(float))/(2*h)))
                    lp,rp,_,_,_,sp=native.forward(plus);lm,rm,_,_,_,sm=native.forward(minus)
                    cp,_,mp=reference(torch.tensor(plus.astype(float)),f);cm,_,mm=reference(torch.tensor(minus.astype(float)),f)
                    smooth=mp['fingerprint']==mm['fingerprint']==meta['fingerprint'] and bool((rp[3]>0).all()) and bool((rm[3]>0).all())
                    rows.append(dict(probe=str(label),expected=expected,native_fd=(lp-lm)/(2*h),cpu_fd=float(cp-cm)/(2*h),
                                     native_plus_loss=lp,native_minus_loss=lm,cpu_plus_loss=float(cp),cpu_minus_loss=float(cm),
                                     plus_samples=sp.tolist(),minus_samples=sm.tolist(),plus_meta=mp,minus_meta=mm,smooth=bool(smooth)))
                diffs.append(dict(step=h,rows=rows,native_fd_relative=relative([r['native_fd'] for r in rows],np.array([r['expected'] for r in rows])),
                                  cpu_fd_relative=relative([r['cpu_fd'] for r in rows],np.array([r['expected'] for r in rows]))))
            n=f['n'];groups=dict(geometry=relative(gpu[:9*n],exact[:9*n]),
                                sigma_opacity=relative(gpu[9*n:11*n],exact[9*n:11*n]),SH=relative(gpu[11*n:],exact[11*n:]))
            forward_error=float(np.max(np.abs(data[-1]-rgb.detach().numpy())))
            smooth=all(r['smooth'] for s in diffs for r in s['rows'])
            gate=all(e<1e-3 for e in groups.values()) and forward_error<3e-5 and smooth \
                and diffs[1]['native_fd_relative']<1e-2 and diffs[1]['cpu_fd_relative']<1e-3
            row=dict(name=scene['name'],variant=variant,gradients_relative=groups,forward_max_abs=forward_error,
                     native_loss=data[0],cpu_loss=float(loss.detach()),native_gradient=gpu.tolist(),exact_gradient=exact.tolist(),
                     metadata=meta,native_radii=data[1][3].cpu().tolist(),finite_differences=diffs,all_perturbations_smooth=smooth,gate=bool(gate))
            records.append(row)
            np.savez_compressed(OUT/(variant+'_'+scene['name']+'.npz'),native_rgb=data[1][1].cpu().numpy(),
                native_samples=data[-1],cpu_samples=rgb.detach().numpy(),**{f'native_grad_{j}':v for j,v in enumerate(arrays)})
            (OUT/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps({k:row[k] for k in ['name','variant','gradients_relative','forward_max_abs','all_perturbations_smooth','gate']}),flush=True)
    result=dict(protocol_sha256=sha(OUT/'protocol.json'),records=records,script_sha256=sha(Path(__file__)),
                all_fixed_gates=all(r['gate'] for r in records if r['variant']=='fixed'),
                total_probe_step_pairs=sum(len(s['rows']) for r in records for s in r['finite_differences']),
                training_performed=False,quality_gain_established=False,seconds=time.time()-start)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['all_fixed_gates','total_probe_step_pairs','seconds']}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        if OUT.exists():(OUT/'execution_failure.txt').write_text(traceback.format_exc())
        raise
