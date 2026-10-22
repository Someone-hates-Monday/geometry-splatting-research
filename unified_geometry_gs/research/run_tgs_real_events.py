"""Bounded no-grad split simulations on one provenance-checked real checkpoint.

No optimizer, training loop, or reconstruction-quality claim. Event/camera
selection is frozen before viewing split outcomes; every selected case is kept.
"""
import hashlib
import importlib.util
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from audit_tgs_real_checkpoint import ROOT, INPUT, OUT, load_input
from tgs_native_diagnostic_backend import load_backend


def file_sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def camera_tensors(c,w,h):
    c2w=np.eye(4);c2w[:3,:3]=c['rotation'];c2w[:3,3]=c['position']
    view=torch.tensor(np.linalg.inv(c2w),dtype=torch.float32,device='cuda').T.contiguous()
    tx,ty=c['width']/(2*c['fx']),c['height']/(2*c['fy'])
    p=torch.zeros((4,4),dtype=torch.float32,device='cuda')
    p[0,0]=1/tx;p[1,1]=1/ty;p[2,2]=100/99.99;p[2,3]=-1/99.99;p[3,2]=1
    full=(view.unsqueeze(0).bmm(p.T.unsqueeze(0))).squeeze(0).contiguous()
    center=view.inverse()[3,:3].contiguous()
    return {'view':view,'projection':full,'center':center,'tx':tx,'ty':ty,'w':w,'h':h}


def pack(points,sigma,opacity,sh):
    n=len(points)
    return {'points':points.contiguous(),'sigma':sigma.contiguous(),'opacity':opacity.contiguous(),
            'sh':sh.contiguous(),'counts':torch.full((n,),3,dtype=torch.int32,device='cuda'),
            'offsets':torch.arange(n,dtype=torch.int32,device='cuda')*3,
            'zero':torch.zeros(n,device='cuda'),'white':torch.ones((n,3),device='cuda')}


@torch.no_grad()
def render(mod,state,cam,appearance='sh',vector_ids=None,precomputed=None):
    empty=torch.empty(0,device='cuda')
    colors=state['white'] if appearance=='white' else (empty if precomputed is None else precomputed)
    sh=state['sh'] if appearance=='sh' and precomputed is None else empty
    res=mod.rasterize_triangles(torch.zeros(3,device='cuda'),state['points'],state['sigma'],
        state['counts'],state['offsets'],colors,state['opacity'],state['zero'].clone(),state['zero'].clone(),
        cam['view'],cam['projection'],len(state['points']),cam['tx'],cam['ty'],cam['h'],cam['w'],
        sh,3,cam['center'],False,True)
    torch.cuda.synchronize()
    result={'rgb':res[1].cpu().numpy(),'alpha':res[2][1].cpu().numpy(),
            'depth':res[2][0].cpu().numpy(),'tiles':int(res[0])}
    ids=slice(None) if vector_ids is None else vector_ids
    result.update({'radii':res[3][ids].cpu().numpy(),'inradius':res[8][ids].cpu().numpy(),
                   'max_blend':res[9][ids].cpu().numpy(),'scaling':res[7][ids].cpu().numpy()})
    assert all(np.isfinite(result[k]).all() for k in ['rgb','alpha','depth','inradius'])
    return result


def split(state,index):
    a,b,c=state['points'][index]
    ab,ac,bc=(a+b)/2,(a+c)/2,(b+c)/2
    child=torch.stack([torch.stack([a,ab,ac]),torch.stack([b,ab,bc]),
                       torch.stack([c,ac,bc]),torch.stack([ab,ac,bc])])
    attrs={}
    for key in ['points','sigma','opacity','sh']:
        v=state[key];replacement=child if key=='points' else v[index:index+1].repeat((4,)+(1,)*(v.ndim-1))
        attrs[key]=torch.cat([v[:index],v[index+1:],replacement])
        assert torch.equal(attrs[key][:index],v[:index])
        assert torch.equal(attrs[key][index:len(v)-1],v[index+1:])
        if key!='points':assert torch.equal(attrs[key][-4:],v[index:index+1].expand_as(replacement))
    return pack(attrs['points'],attrs['sigma'],attrs['opacity'],attrs['sh'])


def metrics(before,after,tolerance):
    diff=np.abs(after['rgb'].astype(float)-before['rgb'])
    da=np.abs(after['alpha'].astype(float)-before['alpha'])
    dd=np.abs(after['depth'].astype(float)-before['depth'])
    return {'rgb_max_abs':float(diff.max()),'rgb_L1_sum':float(diff.sum()),
            'rgb_mean_abs':float(diff.mean()),'rgb_pixels_above_tolerance':int((diff.max(0)>tolerance).sum()),
            'rgb_pixels_nonzero':int((diff.max(0)>0).sum()),'alpha_L1_sum':float(da.sum()),
            'depth_max_abs':float(dd.max()),'alpha_max_abs':float(da.max())}


def save_change(path,before,after):
    changed=(np.abs(after['rgb']-before['rgb']).max(0)>0)|(after['alpha']!=before['alpha'])|(after['depth']!=before['depth'])
    y,x=np.nonzero(changed)
    box=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1] if len(x) else [0,0,0,0]
    x0,y0,x1,y1=box
    np.savez_compressed(path,bbox=np.array(box),before_rgb=before['rgb'][:,y0:y1,x0:x1],
        after_rgb=after['rgb'][:,y0:y1,x0:x1],before_alpha=before['alpha'][y0:y1,x0:x1],
        after_alpha=after['alpha'][y0:y1,x0:x1],before_depth=before['depth'][y0:y1,x0:x1],
        after_depth=after['depth'][y0:y1,x0:x1],shape=np.array(before['alpha'].shape))
    return box


def continuous(v,sigma,opacity,cam):
    points=np.asarray(v,dtype=float)
    matrix=cam['projection'].cpu().numpy().T.astype(float)
    hom=np.c_[points,np.ones(3)]@matrix.T
    if np.any(hom[:,3]<=.2):return None
    p=((hom[:,:2]/(hom[:,3:4]+1e-7)+1)*[cam['w'],cam['h']]-1)/2
    sides=np.array([np.linalg.norm(p[(j+1)%3]-p[(j+2)%3]) for j in range(3)])
    area=abs(np.linalg.det(np.stack([p[1]-p[0],p[2]-p[0]])))/2
    if sides.sum()==0 or area<1e-12:return None
    s=sides@p/sides.sum();mu=(p.sum(0)+sigma*s)/(sigma+3)
    mass=2*opacity*area/((sigma+1)*(sigma+2))
    return {'mass':float(mass),'centroid':mu.tolist(),'area':float(area),
            'inradius':float(2*area/sides.sum()),'vertices':p.tolist()}


def main():
    start=time.time()
    assert not (OUT/'results.json').exists(),'Finished run already exists: audit it; do not silently restart.'
    audit=json.loads((OUT/'input_audit.json').read_text());assert audit['passed']
    model,hyper,cfg,cameras,selection=load_input()
    model={k:(v.detach().cuda() if isinstance(v,torch.Tensor) else v) for k,v in model.items()}
    state=pack(model['triangles_points'],.01+torch.exp(model['sigma']),torch.sigmoid(model['opacity']),
               torch.cat([model['features_dc'],model['features_rest']],1))
    assert int(model['active_sh_degree'])==3
    original=load_backend();fixed=load_backend(shfix=True)
    w,h=audit['actual_resolution']
    cams=[camera_tensors(c,w,h) for c in audit['selected_cameras']]
    # The imported numerical SH polynomial is pinned, byte-verified source;
    # no training/loader/setup code is imported.
    manifest=json.loads((INPUT/'source_manifest.json').read_text())
    path=INPUT/'source/utils/sh_utils.py'
    assert file_sha(path)==next(e['sha256'] for e in manifest['files'] if e['path']=='utils/sh_utils.py')
    spec=importlib.util.spec_from_file_location('pinned_tgs_eval_sh',path);shmath=importlib.util.module_from_spec(spec);spec.loader.exec_module(shmath)
    baselines=[];max_radius=np.zeros(len(state['points']),dtype=np.int32);repeat_max=0.;checks=[]
    for ci,(camera,meta) in enumerate(zip(cams,audit['selected_cameras'])):
        a=render(original,state,camera)
        for _ in range(2):
            repeated=render(original,state,camera,vector_ids=slice(0,0))
            repeat_max=max(repeat_max,float(np.abs(a['rgb']-repeated['rgb']).max()))
        b=render(fixed,state,camera,vector_ids=slice(0,0))
        white=render(original,state,camera,'white',vector_ids=slice(0,0))
        directions=state['points'].mean(1)-camera['center'][None]
        directions/=directions.norm(dim=1,keepdim=True)
        assert torch.isfinite(directions).all()
        colors=torch.clamp_min(shmath.eval_sh(3,state['sh'].transpose(1,2),directions)+.5,0).contiguous()
        py=render(original,state,camera,vector_ids=slice(0,0),precomputed=colors)
        error=float(np.abs(b['rgb']-py['rgb']).max())
        assert error<2e-4,'Corrected native and pinned Python degree3 SH differ'
        checks.append({'camera':ci,'fixed_vs_python_precomputed_rgb_max_abs':error,
                       'original_vs_fixed_rgb_max_abs':float(np.abs(a['rgb']-b['rgb']).max())})
        max_radius=np.maximum(max_radius,a['radii'])
        np.savez_compressed(OUT/f'baseline_{ci}.npz',original_rgb=a['rgb'],fixed_rgb=b['rgb'],
            white_alpha=white['alpha'],depth=a['depth'],radii=a['radii'],inradius=a['inradius'],
            max_blend=a['max_blend'],view=camera['view'].cpu().numpy(),
            projection=camera['projection'].cpu().numpy(),camera_center=camera['center'].cpu().numpy())
        baselines.append({'original':a,'fixed':b,'white':white})
        print(json.dumps({'stage':'baseline','camera':ci,'image':meta['img_name'],'SH_check':error}),flush=True)
    tolerance=max(3e-5,100*repeat_max)
    eligible=np.flatnonzero((max_radius>24)&(state['opacity'].cpu().numpy().reshape(-1)>=.01))
    key=lambda i:hashlib.sha256(('tgs_real_event_v1/event/'+str(int(i))).encode()).hexdigest()
    ids=sorted(eligible,key=key)[:8]
    assert len(ids)==8
    plan={'event_ids':[int(x) for x in ids],'eligible_count':len(eligible),'eligible_ids_sha256':hashlib.sha256(eligible.astype('<i8').tobytes()).hexdigest(),
          'selection_rule':audit['event_selection_rule'],'event_label':audit['event_label'],
          'repeat_max_abs':repeat_max,'rgb_tolerance':tolerance,'baseline_SH_checks':checks,
          'source_audit_sha256':file_sha(OUT/'input_audit.json'),'saved_before_split_rendering':True}
    (OUT/'event_selection.json').write_text(json.dumps(plan,indent=2)+'\n')
    np.save(OUT/'eligible_ids.npy',eligible)
    rows=[]
    for ei,index in enumerate(ids):
        after_state=split(state,int(index));n=len(after_state['points'])
        point=state['points'][index].cpu().numpy();children=after_state['points'][-4:].cpu().numpy()
        sigma=float(state['sigma'][index]);opacity=float(state['opacity'][index])
        np.savez(OUT/f'event_{ei}_input.npz',parent_id=np.array(index),points=point,children=children,
                 sigma=np.array(sigma),opacity=np.array(opacity),sh=state['sh'][index].cpu().numpy())
        for ci,cam in enumerate(cams):
            row={'event_index':ei,'parent_id':int(index),'camera':ci,'metrics':{},'boxes':{}}
            for mode,mod,appearance in [('original',original,'sh'),('fixed',fixed,'sh'),('white',original,'white')]:
                after=render(mod,after_state,cam,appearance,vector_ids=slice(n-4,n))
                before=baselines[ci][mode]
                row['metrics'][mode]=metrics(before,after,tolerance)
                row['boxes'][mode]=save_change(OUT/f'event_{ei}_camera_{ci}_{mode}.npz',before,after)
                if mode=='original':
                    row.update({'parent_radii':int(before['radii'][index]),
                        'parent_inradius':float(before['inradius'][index]),'parent_max_blend':float(before['max_blend'][index]),
                        'child_radii':after['radii'].tolist(),'child_inradius':after['inradius'].tolist()})
            row['continuous_parent']=continuous(point,sigma,opacity,cam)
            row['continuous_children']=[continuous(x,sigma,opacity,cam) for x in children]
            # Record center-depth ordering changes without claiming a causal
            # separation of sorting from occlusion/window changes.
            vm=cam['view'].cpu().numpy().T
            zp=float((np.r_[point.mean(0),1]@vm.T)[2])
            zc=(np.c_[children.mean(1),np.ones(4)]@vm.T)[:,2]
            row['center_depth']={'parent':zp,'children':zc.tolist()}
            rows.append(row)
            (OUT/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({'stage':'split','event':ei,'parent_id':int(index),'camera':ci,
                              'rgb_max':row['metrics']['original']['rgb_max_abs'],
                              'pixels':row['metrics']['original']['rgb_pixels_above_tolerance']}),flush=True)
        del after_state
    report={'complete':True,'scene':audit['scene'],'iteration':audit['iteration'],'triangles':audit['triangles'],
            'event_label':audit['event_label'],'cases':rows,'selection':plan,
            'no_training':True,'native_original_sha256':file_sha(Path(original.__file__)),
            'native_fixed_sha256':file_sha(Path(fixed.__file__)),'runtime_seconds':time.time()-start,
            'scope':'one real published checkpoint; offline simulations; no captured events or training benefit',
            'sorting_and_occlusion_causally_separated':False}
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':True,'cases':len(rows),'seconds':report['runtime_seconds']}),flush=True)


if __name__=='__main__':
    try:
        with torch.no_grad():main()
    except Exception:
        OUT.mkdir(parents=True,exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc())
        raise
