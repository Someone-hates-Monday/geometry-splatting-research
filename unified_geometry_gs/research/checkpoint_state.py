"""Research-only complete checkpoint state and corrected optimizer binding.

The original capture tuple omits filter_3D, absolute densification statistics,
camera queue and RNG states. Never claim an ordinary .pth alone reproduces a run.
"""
from __future__ import annotations
import hashlib
import json
import random
from enum import Enum
import numpy as np
import torch


def restore_bound(self,model_args,training_args):
    (self.active_sh_degree,self.active_sg_degree,self._xyz,self._features_dc,
     self._features_rest,self._scaling,self._rotation,self._opacity,
     self._sg_axis,self._sg_sharpness,self._sg_color,self.max_radii2D,
     xyz_gradient_accum,denom,opt_dict,self.spatial_lr_scale,self.app_model,
     app_dict,appearance_embeddings)=model_args
    # Bind restored appearance objects BEFORE constructing/loading the optimizer.
    self._appearance_embeddings=appearance_embeddings
    if self.app_model==self.App_model.GOF:
        from scene.appearance_network import AppearanceNetwork
        self.appearance_network=AppearanceNetwork(3+64,3).cuda()
        self.appearance_network.load_state_dict(app_dict)
    self.training_setup(training_args)
    self.xyz_gradient_accum=xyz_gradient_accum
    self.denom=denom
    self.optimizer.load_state_dict(opt_dict)
    verify_binding(self)


def verify_binding(gs):
    expected={'xyz':gs._xyz,'f_dc':gs._features_dc,'f_rest':gs._features_rest,
        'opacity':gs._opacity,'scaling':gs._scaling,'rotation':gs._rotation,
        'sg_axis':gs._sg_axis,'sg_sharpness':gs._sg_sharpness,'sg_color':gs._sg_color,
        'appearance_embeddings':gs._appearance_embeddings}
    for group in gs.optimizer.param_groups:
        name=group['name']
        if name=='appearance_network':
            if [id(p) for p in group['params']]!=[id(p) for p in gs.appearance_network.parameters()]:
                raise ValueError('Stale appearance network optimizer binding.')
        elif group['params'][0] is not expected[name]:
            raise ValueError(f'Stale optimizer parameter binding: {name}')
    return True


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
        'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])


def supplemental(gs,scene,stack):
    return {'filter_3D':gs.filter_3D.detach().clone(),
        'xyz_gradient_accum_abs':gs.xyz_gradient_accum_abs.detach().clone(),
        'percent_dense':gs.percent_dense,
        'train_camera_order':[(c.image_name,c.uid) for c in scene.getTrainCameras()],
        'camera_queue':None if stack is None else [(c.image_name,c.uid) for c in stack],
        'rng':rng_state()}


def apply_supplemental(gs,scene,supp):
    gs.filter_3D=supp['filter_3D'].to(gs._xyz.device)
    gs.xyz_gradient_accum_abs=supp['xyz_gradient_accum_abs'].to(gs._xyz.device)
    gs.percent_dense=supp['percent_dense']
    lookup={(c.image_name,c.uid):c for c in scene.getTrainCameras()}
    wanted=[tuple(k) for k in supp['train_camera_order']]
    if set(lookup)!=set(wanted) or len(lookup)!=len(wanted):
        raise ValueError('Camera identities changed.')
    scene.train_cameras[1.0]=[lookup[k] for k in wanted]
    stack=None if supp['camera_queue'] is None else [lookup[tuple(k)] for k in supp['camera_queue']]
    restore_rng(supp['rng'])
    return stack


def state_digest(value):
    """Canonical content digest; excludes Python object identities and file paths."""
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous()
            h.update(b'T'+str(y.dtype).encode()+str(tuple(y.shape)).encode())
            h.update(y.numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(b'N'+str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
        elif isinstance(x,dict):
            h.update(b'D')
            for k in sorted(x,key=lambda k:(type(k).__name__,str(k))):
                visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(b'L'+str(len(x)).encode())
            for y in x:visit(y)
        elif isinstance(x,Enum):
            h.update(b'E'+type(x).__qualname__.encode())
            visit(x.value)
        elif isinstance(x,np.generic):
            visit(x.item())
        else:
            h.update(b'V'+json.dumps(x,sort_keys=True,allow_nan=False).encode()+b'\0')
    visit(value)
    return h.hexdigest()
