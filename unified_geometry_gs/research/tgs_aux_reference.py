"""Auxiliary TGS maps on fixed contributor branches, independent Torch graph."""
import copy
import numpy as np
import torch

from tgs_mixed_reference import Native, reference as rgb_reference, unpack


def reference(theta,f):
    p,sigma,opacity,sh=unpack(theta,f)
    n=len(p);S=len(f['pixels'])
    const=lambda x:torch.tensor(np.asarray(x),dtype=torch.float64)
    alpha=[];raw=[]
    for i in range(n):
        single=dict(f,n=1,degree=-1,colors=np.ones((1,3)),background=np.zeros(3),pixel_grad=np.zeros((S,3)))
        single.pop('preprocess_visible',None)
        t=torch.cat([p[i].reshape(-1),sigma[i:i+1],opacity[i:i+1]])
        _,rgb,meta=rgb_reference(t,single)
        alpha.append(rgb[:,0]);raw.append(meta['fingerprint'])
    alpha=torch.stack(alpha)
    view=const(f['view'])
    center=torch.cat([p.mean(1),torch.ones((n,1),dtype=torch.float64)],-1)@view.T
    depths=center[:,2]
    cross=torch.linalg.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])@view[:3,:3].T
    length=torch.linalg.vector_norm(cross,dim=-1)
    normals=cross/length.clamp_min(1e-4)[:,None]
    facing=(normals*center[:,:3]).sum(-1)
    sign=torch.where(facing>0,-1.,1.)
    normals=normals*sign[:,None]
    order=depths.detach().argsort().tolist()
    T=torch.ones(S,dtype=torch.float64)
    depth=torch.zeros(S,dtype=torch.float64)
    normal=torch.zeros((S,3),dtype=torch.float64)
    median=torch.zeros(S,dtype=torch.float64)
    distortion=torch.zeros(S,dtype=torch.float64)
    M1=torch.zeros(S,dtype=torch.float64);M2=torch.zeros_like(M1)
    done=torch.zeros(S,dtype=torch.bool)
    fp=dict(single=raw,order=order,normal_floor=(length<=1e-4).tolist(),
            normal_flip=(facing>0).tolist(),included=[],terminated=[],median_picks=[])
    for i in order:
        valid=alpha[i]>0
        stop=valid&(T*(1-alpha[i])<1e-4)&~done
        take=valid&~done&~stop
        a=torch.where(take,alpha[i],torch.zeros_like(T));weight=T*a
        m=100/99.8*(1-.2/depths[i])
        distortion=distortion+(m*m*(1-T)+M2-2*m*M1)*weight
        M1=M1+m*weight;M2=M2+m*m*weight
        depth=depth+depths[i]*weight
        normal=normal+weight[:,None]*normals[i]
        choose=take&(T>.5)
        median=torch.where(choose,depths[i],median)
        T=T*(1-a);done=done|stop
        fp['included'].append(take.tolist());fp['terminated'].append(stop.tolist());fp['median_picks'].append(choose.tolist())
    aux=torch.stack([depth,1-T,normal[:,0],normal[:,1],normal[:,2],median,distortion],-1)
    loss=(aux*const(f['aux_grad'])).sum()
    # This diagnostic intentionally has no SH appearance or RGB contribution.
    loss=loss+theta.sum()*0
    meta=dict(fingerprint=fp,normal_lengths=length.detach().tolist(),normal_maps=normals.detach().tolist(),
              triangle_alpha=alpha.detach().tolist(),depth=depths.detach().tolist())
    return loss,aux,meta


class AuxNative(Native):
    def __init__(self,mod,f):
        super().__init__(mod,f)
        a=np.zeros((7,f['h'],f['w']),dtype=np.float32)
        for (x,y),g in zip(f['pixels'],f['aux_grad']):a[:,y,x]=g
        self.grad_others=self.t(a)

    def forward(self,theta):
        old=super().forward(theta)
        aux=np.array([old[1][2][:,y,x].cpu().numpy() for x,y in self.f['pixels']])
        loss=float((aux.astype(float)*self.f['aux_grad'].astype(float)).sum())
        return (loss,*old[1:5],aux)
