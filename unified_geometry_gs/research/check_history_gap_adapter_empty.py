"""Saved actual-ray replay fixture checks mixed/empty normalization exactly."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,math
import numpy as np
import torch
from research.history_gap_live_adapter import Capture,LiveAdapter,load_native,ROOT
R=ROOT/'research';BASE=R/'experiments/history_gap_adapter_v3';OUT=R/'results/history_gap_adapter_empty_v1.json'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    assert not OUT.exists();torch.set_num_threads(1);p=BASE/'scan37_0000_scatter.npz';m=read(R/'experiments/history_gap_warp_v2/manifest.json')['binary'];native=load_native(ROOT/m['path'],m['sha256'])
    with np.load(p) as d:source=d['source'];raw=d['raw'];ids=d['ids'];sections=d['sections']
    unique,first,inverse=np.unique(ids,return_index=True,return_inverse=True);src=source[first];rr=raw[first];assert np.array_equal(source[:,3:10],src[inverse,3:10]) and np.array_equal(source[:,35],src[inverse,35]);N=len(unique);M=len(sections)-1;device='cuda'
    gs=SimpleNamespace(_xyz=torch.nn.Parameter(torch.tensor(rr[:,:3],dtype=torch.float32,device=device)),_scaling=torch.nn.Parameter(torch.tensor(rr[:,3:6],dtype=torch.float32,device=device)),_rotation=torch.nn.Parameter(torch.tensor(rr[:,6:10],dtype=torch.float32,device=device)),_opacity=torch.nn.Parameter(torch.tensor(rr[:,10:11],dtype=torch.float32,device=device)),filter_3D=torch.tensor(rr[:,11:12],dtype=torch.float32,device=device))
    q=np.concatenate([source[sections[:-1],32:34],np.zeros((1,2))]);view=torch.eye(4,dtype=torch.float32,device=device);view[:3,:3]=torch.tensor(source[0,20:29].reshape(3,3).T,dtype=torch.float32,device=device);view[3,:3]=torch.tensor(source[0,29:32],dtype=torch.float32,device=device)
    lengths=np.concatenate([np.diff(sections),[0]]);offsets=np.concatenate([sections,[sections[-1]]]);records=np.stack([inverse,np.zeros_like(inverse)],1);meta=read(R/'experiments/world_root_vjp_v1/scan37_0000.json')['meta'];metadata=np.array([meta[k] for k in ['fx','fy','limx','limy','kernel','floor']]);cap=Capture(torch.tensor(src[:,:3],dtype=torch.float32,device=device),torch.tensor(src[:,3:6],dtype=torch.float32,device=device),torch.tensor(src[:,6:10],dtype=torch.float32,device=device),view,torch.tensor(q,dtype=torch.float32,device=device),torch.tensor(src[:,35],dtype=torch.float32,device=device),torch.tensor(offsets,dtype=torch.int64,device=device),torch.tensor(records,dtype=torch.int32,device=device),torch.tensor(lengths,dtype=torch.int64,device=device),metadata,{})
    adapter=LiveAdapter(None,None,native);parameters=[gs._xyz,gs._scaling,gs._rotation,gs._opacity];rows=[];grad=[];losses=[]
    for name,pixels in [('nonempty',np.arange(M)),('mixed',np.arange(M+1)),('all_empty',np.array([M]))]:
        for parameter in parameters:parameter.grad=None
        loss,detail=adapter.loss(gs,cap,pixels);loss.backward();g=torch.cat([torch.zeros_like(v) if v.grad is None else v.grad for v in parameters],1).cpu().numpy();grad.append(g);losses.append(float(loss.detach()));rows.append(dict(name=name,samples=detail['sample_count'],empty=detail['empty'],loss=float(loss.detach()),finite_gradients=bool(np.isfinite(g).all())))
    expected=(losses[0]*M+math.log(4)**2)/(M+1);relative=float(np.linalg.norm(grad[1]-grad[0]*M/(M+1))/max(np.linalg.norm(grad[0]*M/(M+1)),1e-30));assert abs(losses[1]-expected)<1e-14 and relative<1e-5 and np.count_nonzero(grad[2])==0 and losses[2]==math.log(4)**2
    np.savez_compressed(R/'results/history_gap_adapter_empty_v1.npz',nonempty_gradient=grad[0],mixed_gradient=grad[1],all_empty_gradient=grad[2],losses=np.array(losses),M=np.array([M]));result=dict(status='complete',gate_pass=True,source_sha256={q.relative_to(ROOT).as_posix():sha(q) for q in [Path(__file__),R/'history_gap_live_adapter.py',p]},fixture='Repacked saved actual scan37 rays on their actual unique raw inputs; an appended empty ray tests the adapter only, not new native capture.',rows=rows,mixed_loss_error=abs(losses[1]-expected),mixed_gradient_relative_error=relative,all_empty_gradient_exactly_zero=True);OUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
