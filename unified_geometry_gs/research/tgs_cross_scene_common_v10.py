"""Read-only v9/v5 modules and safely loaded new real checkpoints; no optimizer."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from audit_tgs_real_checkpoint import cfg_parse
from benchmark_tgs_backward import Workload
from run_tgs_real_events import camera_tensors
from tgs_window_candidates_common import decode,state,strict_groups

ROOT=Path(__file__).resolve().parent;SOURCE=ROOT/'sources/tgs_cross_scene_v10';OUT=ROOT/'experiments/tgs_cross_scene_v10'
V9=ROOT/'experiments/tgs_window_candidates_v9';VARIANTS=['oracle','coarse','oracle_diag','coarse_diag']
V9_SHA='489230fa253ed3eca2e3a9a5ad1c5458bee2f3489f04a4337ee8c818fa7bb22c'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()

def model_input(scene):
    folder=SOURCE/scene;model_path=folder/'point_cloud/iteration_7000/point_cloud_state_dict.pt'
    model=torch.load(model_path,map_location='cpu',weights_only=True);hp=folder/'point_cloud/iteration_7000/hyperparameters.pt';unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(hp);assert set(unsafe)<={'numpy._core.multiarray.scalar','numpy.core.multiarray.scalar','numpy.dtype'}
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,type(np.dtype('float64')),type(np.dtype('float32'))]):hyper=torch.load(hp,map_location='cpu',weights_only=True)
    cfg=cfg_parse((folder/'cfg_args').read_text());cameras=json.loads((folder/'cameras.json').read_text());return model,hyper,cfg,cameras

def modules():
    assert sha(V9/'evidence_manifest.json')==V9_SHA
    assert json.loads((V9/'precision_audit.json').read_text())['candidate_gates']=={'coarse':True,'standard':False}
    mods={}
    for v in VARIANTS:
        directory=V9 if v.startswith('coarse') else ROOT/'experiments/tgs_phase_cost_v5';r=json.loads((directory/v/'build_receipt.json').read_text());path=Path(r['module_file']);assert sha(path)==r['module_sha256'];s=importlib.util.spec_from_file_location(path.stem,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);mods[v]=m
    return mods

class CrossWorkload(Workload):
    def __init__(self,scene):
        audit=json.loads((OUT/'input_audit.json').read_text());row=next(r for r in audit['scenes'] if r['scene']==scene);model,_,_,_=model_input(scene)
        self.points=model['triangles_points'].detach().cuda().contiguous();self.sigma=(.01+torch.exp(model['sigma'].detach().cuda())).contiguous();self.opacity=torch.sigmoid(model['opacity'].detach().cuda()).contiguous();self.sh=torch.cat([model['features_dc'].detach(),model['features_rest'].detach()],1).cuda().contiguous();self.n=len(self.points)
        self.counts=torch.full((self.n,),3,dtype=torch.int32,device='cuda');self.offsets=torch.arange(self.n,dtype=torch.int32,device='cuda')*3;self.scaling=torch.zeros(self.n,device='cuda');self.density=torch.zeros_like(self.scaling);assert self.scaling.data_ptr()!=self.density.data_ptr();self.empty=torch.empty(0,device='cuda');self.bg=torch.zeros(3,device='cuda')
        self.camera_metadata=row['selected_cameras'];w,h=row['actual_resolution'];self.cameras=[camera_tensors(c,w,h) for c in self.camera_metadata]
        yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij');spatial=1+.25*torch.sin(xx/17)+.2*torch.cos(yy/23);self.grad=(torch.tensor([1.,-.3,.5],device='cuda')[:,None,None]*spatial[None]/(w*h)).contiguous();self.grad_others=torch.zeros((7,h,w),device='cuda');self.rgb_up=self.grad.clone();self.depth_up=torch.ones_like(self.grad_others[0])/(w*h)
        self.checkpoint=SOURCE/scene/'point_cloud/iteration_7000/point_cloud_state_dict.pt';self.metadata=dict(scene=scene,triangles=self.n,resolution=[w,h],sh_degree=3,checkpoint_sha256=sha(self.checkpoint),sigma_activation='0.01+exp',mask_semantics='effective inference mask1 per official load; no saved training mask',upstream='fixed dense RGB and separate uniform weighted-depth; no target/loss optimization')
    def set_loss(self,loss):
        self.grad.copy_(self.rgb_up if loss=='rgb' else torch.zeros_like(self.rgb_up));self.grad_others.zero_()
        if loss=='depth':self.grad_others[0].copy_(self.depth_up)
    def input_hashes(self):
        d={k:thash(getattr(self,k)) for k in ['points','sigma','opacity','sh','counts','offsets','rgb_up','depth_up']}
        for i,c in enumerate(self.cameras):
            for key in ['view','projection','center']:d[f'camera_{i}_{key}']=thash(c[key])
        return d
