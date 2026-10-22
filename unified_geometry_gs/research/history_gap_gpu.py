"""CUDA extended-loss kernel with a portable source-preserving raw graph."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.cpp_extension import load

def build(directory):
    source=Path(__file__).with_suffix('.cu');digest=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    return load(name='history_gap_'+digest,sources=[str(source)],build_directory=str(directory),extra_cuda_cflags=['-O3','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','--allow-unsupported-compiler'],extra_cflags=['-O3'],verbose=False)

def rotation(q):
    w,x,y,z=q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1).reshape(-1,3,3)

def inverse3(R):
    a,b,c=R.unbind(-1);bc=torch.linalg.cross(b,c,dim=-1)
    return torch.stack([bc,torch.linalg.cross(c,a,dim=-1),torch.linalg.cross(a,b,dim=-1)],1)/(a*bc).sum(-1)[:,None,None]

class _Gap(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,rho,E,A,sections,ray_ids,native):
        loss,seeds,info=native.solve(p,rho,E,A,sections)
        ctx.save_for_backward(seeds,ray_ids);ctx.mark_non_differentiable(info)
        return loss,info
    @staticmethod
    def backward(ctx,grad_loss,grad_info):
        seeds,ray_ids=ctx.saved_tensors;g=seeds*grad_loss[ray_ids,None]
        return g[:,0],g[:,1],g[:,2],g[:,3],None,None,None

class Engine:
    def __init__(self,raw,source,sections,metadata,native,device='cuda'):
        self.native=native;self.raw=torch.as_tensor(raw,dtype=torch.float64,device=device);self.source=torch.as_tensor(source,dtype=torch.float64,device=device)
        self.sections=torch.as_tensor(sections,dtype=torch.int64,device=device);self.ids=torch.repeat_interleave(torch.arange(len(sections)-1,device=device),self.sections[1:]-self.sections[:-1]);self.base=self.raw[:,:11]
        src=self.source;first=self.sections[:-1];self.A=src[first,20:29].reshape(-1,3,3);self.translation=src[first,29:32];query=src[first,32:35];Ai=torch.linalg.inv(self.A)
        self.origin=-(Ai@self.translation[:,:,None]).squeeze(-1);self.direction=(Ai@query[:,:,None]).squeeze(-1)/torch.linalg.vector_norm(query,dim=1)[:,None]
        self.meta=torch.as_tensor(metadata,dtype=torch.float64,device=device);self.radius=self.raw[:,11]
        with torch.no_grad():self.gamma0=self.screen(self.base[:,:3],rotation(src[:,6:10]),src[:,3:6]).detach()
    def screen(self,mu,R,scales):
        ids=self.ids;A=self.A[ids];meta=self.meta[ids];cam=(A@mu[:,:,None]).squeeze(-1)+self.translation[ids];z=cam[:,2];uc=torch.minimum(meta[:,2],torch.maximum(-meta[:,2],cam[:,0]/z));vc=torch.minimum(meta[:,3],torch.maximum(-meta[:,3],cam[:,1]/z));zero=torch.zeros_like(z)
        J=torch.stack([meta[:,0]/z,zero,-meta[:,0]*uc/z,zero,meta[:,1]/z,-meta[:,1]*vc/z],1).reshape(-1,2,3)
        projected=torch.diag_embed(scales)@R.transpose(1,2)@A.transpose(1,2)@J.transpose(1,2);c=projected.transpose(1,2)@projected
        det=torch.linalg.cross(projected[:,:,0],projected[:,:,1],dim=1).square().sum(1);det1=det+meta[:,4]*(c[:,0,0]+c[:,1,1])+meta[:,4]**2;floor=meta[:,5]
        return .5*(torch.maximum(det,floor).log()-torch.maximum(det1,floor).log())
    def fields(self,theta):
        base=self.base;src=self.source;variance=theta[:,3:6].exp().square();v0=base[:,3:6].exp().square();f=variance+self.radius[:,None]**2;f0=v0+self.radius[:,None]**2
        scales=src[:,3:6]*torch.sqrt(f/f0);q=src[:,6:10]+(torch.nn.functional.normalize(theta[:,6:10],dim=1)-torch.nn.functional.normalize(base[:,6:10],dim=1));R=rotation(q)
        factor=torch.diag_embed(scales.reciprocal())@inverse3(R);direction=self.direction[self.ids];diff=theta[:,:3]-self.origin[self.ids]
        u=(factor@direction[:,:,None]).squeeze(-1);v=(factor@diff[:,:,None]).squeeze(-1);den=u.square().sum(1);peak=(u*v).sum(1)/den;res=v-peak[:,None]*u;E=res.square().sum(1);rho=den.sqrt()
        gamma=self.screen(theta[:,:3],R,scales);logfilter=.5*(variance.log()-f.log()).sum(1);logfilter0=.5*(v0.log()-f0.log()).sum(1)
        amplitude=src[:,35]*torch.exp(torch.nn.functional.logsigmoid(theta[:,10])-torch.nn.functional.logsigmoid(base[:,10])+logfilter-logfilter0+gamma-self.gamma0)
        return peak.contiguous(),rho.contiguous(),E.contiguous(),amplitude.contiguous()
    def loss(self,theta):return _Gap.apply(*self.fields(theta),self.sections,self.ids,self.native)

def metadata_array(meta):return [meta[k] for k in ['fx','fy','limx','limy','kernel','floor']]
