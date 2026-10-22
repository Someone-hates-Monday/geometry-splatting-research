"""Versioned fixed-input split/color diagnostic utilities; no builds or optimizer."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,CrossWorkload
from run_tgs_real_events import pack,split
from tgs_sh_isolation_common_v12 import paths,exact
OUT=ROOT/'experiments/tgs_split_color_v14'
HELPER=ROOT/'sources/tgs_real_checkpoint_v1/source/utils/sh_utils.py'
RECEIPTS={
 'native':'experiments/tgs_subdivision_v1/native_build/receipt_9d974f09b721.json',
 'sh_only':'experiments/tgs_subdivision_v1/sh_direction_v1/fixed_build_receipt.json'}
EXPECTED={
 'native':'9d974f09b7217a026aaad3ebc48af16c6a9571fcd17a885711b0517b9d409da1',
 'sh_only':'4fa625a41b94545010a579e9c98eb68fb557023c2429be890eaf20d750b00ef1'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def write_new(p,x):
    with p.open('x') as f:f.write(json.dumps(x,indent=2,allow_nan=False)+'\n')

def module(path):
    spec=importlib.util.spec_from_file_location(Path(path).stem,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result

def frozen_modules(plan):
    result={}
    for name,en in plan['modules'].items():
        assert sha(ROOT/en['receipt'])==en['receipt_sha256']
        assert sha(Path(en['path']))==en['sha256']==EXPECTED[name]
        result[name]=module(en['path'])
    return result

@torch.no_grad()
def render(mod,state,cam,colors=None,child_tail=False):
    empty=torch.empty(0,device='cuda')
    res=mod.rasterize_triangles(torch.zeros(3,device='cuda'),state['points'],state['sigma'],
        state['counts'],state['offsets'],empty if colors is None else colors,state['opacity'],
        state['zero'].clone(),state['zero'].clone(),cam['view'],cam['projection'],len(state['points']),
        cam['tx'],cam['ty'],cam['h'],cam['w'],state['sh'] if colors is None else empty,
        3,cam['center'],False,False)
    torch.cuda.synchronize()
    arrays=dict(rgb=res[1].cpu().numpy(),aux=res[2].cpu().numpy(),radii=res[3].cpu().numpy(),
                maxblend=res[9].cpu().numpy(),contributors=paths(res,cam['w'],cam['h']))
    assert np.isfinite(arrays['rgb']).all() and np.isfinite(arrays['aux']).all()
    child_radius=None
    if child_tail:
        mask=arrays['radii'][-4:]>0
        child_radius=[float(res[8][-4:][i].item()) if mask[i] else None for i in range(4)]
    return arrays,int(res[0]),child_radius

def colors(helper,state,cam):
    d=state['points'].mean(1)-cam['center'][None]
    d/=d.norm(dim=1,keepdim=True)
    assert torch.isfinite(d).all()
    return (helper.eval_sh(3,state['sh'].transpose(1,2),d)+.5).clamp_min_(0).contiguous()

def metrics(g,a,d,shape):
    denominator=3*int(shape[0])*int(shape[1]);out={}
    for name,v in [('noncolor',g),('appearance',a),('total',d)]:
        absval=np.abs(v);out[name]=dict(RMS=float(np.sqrt(np.sum(v*v)/denominator)),
            MAE=float(absval.sum()/denominator),peak=float(absval.max(initial=0)),
            pixels_above_3e_minus5=int((absval.max(axis=0,initial=0)>3e-5).sum()),
            pixels_nonzero=int((absval.max(axis=0,initial=0)>0).sum()))
    cross=float(2*np.sum(g*a)/denominator)
    out['signed_energy_cross_term']=cross
    out['energy_identity_residual']=float(np.sum(d*d)/denominator-
        (np.sum(g*g)/denominator+np.sum(a*a)/denominator+cross))
    out['A_over_G_RMS']=out['appearance']['RMS']/out['noncolor']['RMS'] if out['noncolor']['RMS']>1e-14 else None
    out['appearance_material']=out['appearance']['RMS']>out['noncolor']['RMS'] and out['appearance']['peak']>3e-5
    return out

def save_case(folder,before,hold,moved):
    changed=np.any(hold['rgb']!=before['rgb'],axis=0)|np.any(moved['rgb']!=before['rgb'],axis=0)
    changed|=np.any(hold['aux']!=before['aux'],axis=0)|np.any(moved['aux']!=before['aux'],axis=0)
    yy,xx=np.nonzero(changed)
    box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else [0,0,0,0]
    x0,y0,x1,y1=box
    np.savez_compressed(folder/'changes.npz',bbox=np.array(box),shape=np.array(changed.shape),
        before_rgb=before['rgb'][:,y0:y1,x0:x1],hold_rgb=hold['rgb'][:,y0:y1,x0:x1],
        moved_rgb=moved['rgb'][:,y0:y1,x0:x1],before_aux=before['aux'][:,y0:y1,x0:x1],
        hold_aux=hold['aux'][:,y0:y1,x0:x1],moved_aux=moved['aux'][:,y0:y1,x0:x1])
    for version,array in [('hold',hold),('moved',moved)]:
        for key in ['rgb','aux']:
            reconstructed=before[key].copy();reconstructed[:,y0:y1,x0:x1]=array[key][:,y0:y1,x0:x1]
            assert exact(reconstructed,array[key])
    b=before['rgb'][:,y0:y1,x0:x1].astype(float)
    h=hold['rgb'][:,y0:y1,x0:x1].astype(float);m=moved['rgb'][:,y0:y1,x0:x1].astype(float)
    return box,metrics(h-b,m-h,m-b,changed.shape)
