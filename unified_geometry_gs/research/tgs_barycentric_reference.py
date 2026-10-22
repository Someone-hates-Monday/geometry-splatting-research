"""Independent full-image geometry reference using edge determinants / barycentrics."""
import numpy as np
import torch
from tgs_mixed_reference import sh_basis

def evaluate(theta,f,pixels):
    n=f['n'];p=theta[:9*n].reshape(n,3,3);sigma=theta[9*n:10*n];opacity=theta[10*n:11*n]
    hom=np.concatenate([p,np.ones((n,3,1))],-1)@f['projection'].astype(float).T
    q=(hom[...,:2]/(hom[...,3:]+1e-7)+1)*[f['w']/2,f['h']/2]-.5
    e=np.roll(q,-1,1)-q;length=np.sqrt((e*e).sum(-1));perimeter=length.sum(1)
    delta=q[:,1]-q[:,0];other=q[:,2]-q[:,0];D=delta[:,0]*other[:,1]-delta[:,1]*other[:,0]
    diff=pixels[None,:,None,:]-q[:,None,:,:]
    cross=e[:,None,:,0]*diff[...,1]-e[:,None,:,1]*diff[...,0]
    signed=cross*np.sign(D)[:,None,None]
    inside=(signed>=0).all(-1)
    u=(signed*perimeter[:,None,None]/(np.abs(D)[:,None,None]*length[:,None,:])).min(-1)
    raw=opacity[:,None]*np.where(inside,np.maximum(u,0),1)**sigma[:,None]
    alpha=np.minimum(.99,raw)
    valid=inside&(alpha>=1/255)&np.asarray(f.get('preprocess_visible',np.ones(n)),bool)[:,None]
    alpha=np.where(valid,alpha,0)
    if f['degree']<0:colors=f['colors'].astype(float)
    else:
        sh=theta[11*n:].reshape(n,(f['degree']+1)**2,3)
        d=p.mean(1)-f['campos'].astype(float);d/=np.linalg.norm(d,axis=1,keepdims=True)
        basis=sh_basis(torch.tensor(d),f['degree']).numpy()
        colors=np.maximum(0,np.einsum('nb,nbc->nc',basis,sh)+.5)
    depth=np.c_[p.mean(1),np.ones(n)]@f['view'].astype(float).T
    order=np.argsort(depth[:,2]);T=np.ones(len(pixels));rgb=np.zeros((len(pixels),3));done=np.zeros(len(pixels),bool)
    for i in order:
        stop=valid[i]&~done&(T*(1-alpha[i])<1e-4)
        take=valid[i]&~done&~stop;a=np.where(take,alpha[i],0)
        rgb+=(T*a)[:,None]*colors[i];T*=1-a;done|=stop
    return rgb+T[:,None]*f['background'].astype(float)
