"""CPU-only prevalence profile, without training or claiming GPU branch counters."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from audit_tgs_real_checkpoint import ROOT,load_input

OUT=ROOT/'experiments/tgs_branch_aux_v1'


def profile(mask,visible,contributing,cap_possible):
    return dict(total=int(len(mask)),normal_floor_reference=int(mask.sum()),
        cached_visible=int(visible.sum()),floor_and_cached_visible=int((mask&visible).sum()),
        cached_contributing=int(contributing.sum()),floor_and_cached_contributing=int((mask&contributing).sum()),
        opacity_above_099_and_cached_visible=int((cap_possible&visible).sum()),
        opacity_above_099_and_cached_contributing=int((cap_possible&contributing).sum()))


def main():
    target=OUT/'real_branch_profile.json';assert not target.exists()
    model,hyper,_,_,selection=load_input()
    p=model['triangles_points'].detach().double()
    world_cross=torch.linalg.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
    norm=torch.linalg.vector_norm(world_cross,dim=1).numpy()
    # A separately expressed NumPy cross product checks the area/floor profile.
    q=p.numpy();a=q[:,1]-q[:,0];b=q[:,2]-q[:,0]
    np_cross=np.stack([a[:,1]*b[:,2]-a[:,2]*b[:,1],a[:,2]*b[:,0]-a[:,0]*b[:,2],a[:,0]*b[:,1]-a[:,1]*b[:,0]],-1)
    assert np.max(np.abs(np_cross-world_cross.numpy()))<1e-12
    np_norm=np.sqrt((np_cross*np_cross).sum(-1))
    assert np.array_equal(np_norm<1e-4,norm<1e-4)
    op=torch.sigmoid(model['opacity'].detach()).numpy().ravel()
    cap_possible=op>.99
    audit=json.loads((ROOT/'experiments/tgs_real_event_v1/input_audit.json').read_text())
    rows=[]
    for ci in [2,3]:
        cam=audit['selected_cameras'][ci];assert cam['split']=='train'
        path=ROOT/f'experiments/tgs_real_event_v1/baseline_{ci}.npz'
        with np.load(path) as z:
            visible=z['radii']>0;contributing=z['max_blend']>0
            view=z['view'].T.astype(float)
        cross_view=np_cross@view[:3,:3].T
        view_norm=np.sqrt((cross_view*cross_view).sum(-1))
        floor=view_norm<1e-4
        rows.append(dict(camera=cam['img_name'],**profile(floor,visible,contributing,cap_possible),
                         baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         near_floor_within_10ppm=int((np.abs(view_norm/1e-4-1)<=1e-5).sum())))
    directory=ROOT/'sources/tgs_real_checkpoint_v1'/selection['scene']/'point_cloud'/selection['iteration']
    names=['lambda_normals','lambda_dist','iteration_mesh']
    saved={k:float(hyper[k]) for k in names if k in hyper and isinstance(hyper[k],(int,float,np.number))}
    result=dict(checkpoint_sha256=hashlib.sha256((directory/'point_cloud_state_dict.pt').read_bytes()).hexdigest(),
        scope='double-precision geometric branch profile crossed with cached full-scene visibility/contribution; not instrumented GPU counters',
        total_triangles=len(norm),world_cross_norm_quantiles=np.quantile(norm,[0,.01,.1,.5,.9,.99,1]).tolist(),
        normal_floor_reference=int((norm<1e-4).sum()),opacity_above_099=int(cap_possible.sum()),
        max_opacity=float(op.max()),per_train_camera=rows,
        numpy_and_torch_floor_classification_equal=True,saved_training_fields_found=saved,
        pinned_defaults=dict(lambda_normals=.0001,lambda_dist=0.,iteration_mesh=5000),
        actual_training_configuration_proven=False,
        limitations=['opacity>0.99 is only a necessary condition for a capped real-valued window, not an observed cap frequency',
                     'normal-floor membership does not measure incoming normal-loss gradients or quality impact',
                     'max-edge ties were not counted; no inference from two cameras to a dataset distribution'],
        training=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
