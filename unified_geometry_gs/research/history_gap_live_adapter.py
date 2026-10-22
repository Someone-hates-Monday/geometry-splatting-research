"""Actual native capture and GPU-only sampled shared-parameter loss."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,importlib.util,math,time
import numpy as np
import torch
from research.history_gap_gpu_v2 import Engine
from research.exact_outlier_capture_rebuilt import native_args,parse
from research.benchmark_csr_capture import offsets_for
ROOT=Path(__file__).resolve().parents[1]

def load_native(path,expected):
    path=Path(path)
    with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected,str(path)
    spec=importlib.util.spec_from_file_location(path.stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def measured(fn):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);start=time.perf_counter();a.record();result=fn();b.record();b.synchronize();return result,dict(event_ms=float(a.elapsed_time(b)),wall_ms=1000*(time.perf_counter()-start))
def camera_metadata(args):
    tx=np.float32(args[16]);ty=np.float32(args[17]);W=np.float32(args[20]);H=np.float32(args[19])
    return np.array([np.float32(W/(np.float32(2)*tx)),np.float32(H/(np.float32(2)*ty)),np.float32(np.float32(1.3)*tx),np.float32(np.float32(1.3)*ty),np.float32(args[18]),np.float32(1e-6)],dtype=np.float64)
@dataclass
class Capture:
    xyz:torch.Tensor
    sigma:torch.Tensor
    quaternion:torch.Tensor
    view:torch.Tensor
    query:torch.Tensor
    amplitude:torch.Tensor
    offsets:torch.Tensor
    records:torch.Tensor
    counts:torch.Tensor
    metadata:np.ndarray
    times:dict
    outputs:object=None
class LiveAdapter:
    def __init__(self,query_native,csr_native,loss_native):self.query_native=query_native;self.csr_native=csr_native;self.loss_native=loss_native
    def capture(self,cam,gs,pipe,bg,kernel,keep_outputs=False):
        torch.cuda.synchronize();start=time.perf_counter();times={}
        with torch.no_grad():
            args=native_args(cam,gs,pipe,bg,kernel);q,times['query_count']=measured(lambda:self.query_native.rasterize_gaussians(*args));counts=q[4][3].to(torch.int64)
            (offsets,records,status),times['csr_scan_allocate']=measured(lambda:offsets_for(counts));c,times['csr_capture']=measured(lambda:self.csr_native.rasterize_gaussians(*args,offsets,records,status))
            if bool(torch.any(status!=0)):raise RuntimeError('Native CSR write/count failed')
            if not torch.equal(c[4][3],q[4][3]):raise RuntimeError('Query/CSR counts differ')
            parsed,_=parse(q,'affine',args[1].shape[0],cam.image_width,cam.image_height);amp=parsed['geom']['conic'][:,3].contiguous();query=q[4][6:8].permute(1,2,0).reshape(-1,2).contiguous();torch.cuda.synchronize();times['capture_total_wall_ms']=1000*(time.perf_counter()-start)
            return Capture(args[1].detach(),args[4].detach(),args[5].detach(),args[14].detach(),query,amp,offsets,records,counts.reshape(-1),camera_metadata(args),times,(q,c) if keep_outputs else None)
    def loss(self,gs,capture,pixels):
        torch.cuda.synchronize();start=time.perf_counter();pixels=torch.as_tensor(pixels,device='cuda',dtype=torch.int64);M=pixels.numel()
        if M==0:raise ValueError('Nonempty sample required')
        counts=capture.counts[pixels];nonempty=counts>0;active_pixels=pixels[nonempty];lengths=counts[nonempty];empty=M-active_pixels.numel();constant=empty*math.log(4)**2
        if active_pixels.numel()==0:
            loss=gs._xyz[0,0].double()*0+constant/M;torch.cuda.synchronize();return loss,dict(sample_pixels=pixels,active_pixels=active_pixels,empty=empty,sample_count=M,node_count=0,prepare_loss_wall_ms=1000*(time.perf_counter()-start),info=torch.empty((0,8),device='cuda',dtype=torch.float64))
        sections=torch.cat([torch.zeros(1,device='cuda',dtype=torch.int64),torch.cumsum(lengths,0)]);ray=torch.repeat_interleave(torch.arange(len(lengths),device='cuda'),lengths);local=torch.arange(ray.numel(),device='cuda')-sections[ray];positions=capture.offsets[active_pixels][ray]+local;ids=capture.records[positions,0].long()
        if bool(torch.any(ids<0)) or bool(torch.any(ids>=gs._xyz.shape[0])):raise RuntimeError('Invalid captured primitive ID')
        theta=torch.cat([gs._xyz[ids],gs._scaling[ids],gs._rotation[ids],gs._opacity[ids]],1).double();raw=torch.cat([theta.detach(),gs.filter_3D[ids].detach().double()],1)
        with torch.no_grad():
            source=torch.zeros((ids.numel(),36),device='cuda',dtype=torch.float64);source[:,:3]=capture.xyz[ids].double();source[:,3:6]=capture.sigma[ids].double();source[:,6:10]=capture.quaternion[ids].double();source[:,10]=1;source[:,20:29]=capture.view[:3,:3].T.reshape(1,9).double();source[:,29:32]=capture.view[3,:3].reshape(1,3).double();source[:,32:34]=capture.query[active_pixels][ray].double();source[:,34]=1;source[:,35]=capture.amplitude[ids].double()
        metadata=np.repeat(capture.metadata[None],active_pixels.numel(),axis=0);engine=Engine(raw,source,sections,metadata,self.loss_native);values,info=engine.loss(theta)
        if not bool(torch.isfinite(values).all()) or not bool((info[:,0]>0).all()):raise FloatingPointError('Loss native failed; do not update optimizer')
        loss=(values.sum()+constant)/M;torch.cuda.synchronize()
        return loss,dict(sample_pixels=pixels,active_pixels=active_pixels,empty=empty,sample_count=M,node_count=ids.numel(),ids=ids,sections=sections,source=source,raw=raw,values=values,info=info,prepare_loss_wall_ms=1000*(time.perf_counter()-start))
