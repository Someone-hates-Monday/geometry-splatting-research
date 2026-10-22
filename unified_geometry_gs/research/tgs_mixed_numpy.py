"""Independent NumPy complex-step reference; no Torch, no native code import.

Uses signed area/perimeter, not the Torch incenter expression. Decisions use
real parts, preserving the base smooth branch under imaginary perturbations.
Norms deliberately use sqrt(sum(z*z)), with no complex conjugation.
SH real polynomial convention/constants follow the archived author's source.
"""
import numpy as np


def evaluate(theta,f):
    n=int(f['n']);degree=int(f['degree'])
    p=theta[:9*n].reshape(n,3,3)
    sig,op=theta[9*n:10*n],theta[10*n:11*n]
    sh=theta[11*n:].reshape(n,(degree+1)**2,3) if degree>=0 else None
    h=np.concatenate([p,np.ones((n,3,1))],-1)@f['projection'].astype(float).T
    xy=(h[...,:2]/(h[...,3:]+1e-7)+1)*np.array([f['w']/2,f['h']/2])-.5
    edges=np.roll(xy,-1,1)-xy
    length=np.sqrt((edges*edges).sum(-1))
    D=(xy[:,1,0]-xy[:,0,0])*(xy[:,2,1]-xy[:,0,1])-(xy[:,1,1]-xy[:,0,1])*(xy[:,2,0]-xy[:,0,0])
    sign=np.sign(D.real)
    radius=sign*D/length.sum(1)
    normals=np.stack([edges[...,1],-edges[...,0]],-1)/length[...,None]*sign[:,None,None]
    offsets=-(normals*xy).sum(-1)
    all_phi=np.einsum('sd,nkd->nsk',f['pixels'],normals)+offsets[:,None]
    active=all_phi.real.argmax(-1)
    phi=np.take_along_axis(all_phi,active[...,None],axis=-1)[...,0]
    inside=phi.real<=0
    u=np.where(inside,-phi/radius[:,None],1)
    raw_alpha=op[:,None]*u**sig[:,None]
    cap=raw_alpha.real>=.99
    alpha=np.where(cap,.99,raw_alpha)
    valid=inside&(alpha.real>=1/255)&np.asarray(f.get('preprocess_visible',np.ones(n)),dtype=bool)[:,None]
    alpha=np.where(valid,alpha,0)
    if sh is None:
        raw_colors=f['colors'].astype(float)
    else:
        d=p.mean(1)-f['campos'].astype(float)
        d=d/np.sqrt((d*d).sum(1))[:,None]
        x,y,z=d.T
        b=[x*0+.28209479177387814]
        if degree>=1:
            b.extend([-.4886025119029199*y,.4886025119029199*z,-.4886025119029199*x])
        if degree>=2:
            b.extend([1.0925484305920792*x*y,-1.0925484305920792*y*z,
                .31539156525252005*(2*z*z-x*x-y*y),-1.0925484305920792*x*z,.5462742152960396*(x*x-y*y)])
        if degree>=3:
            b.extend([-.5900435899266435*y*(3*x*x-y*y),2.890611442640554*x*y*z,
                -.4570457994644658*y*(4*z*z-x*x-y*y),.3731763325901154*z*(2*z*z-3*x*x-3*y*y),
                -.4570457994644658*x*(4*z*z-x*x-y*y),1.445305721320277*z*(x*x-y*y),
                -.5900435899266435*x*(x*x-3*y*y)])
        raw_colors=np.einsum('nb,nbc->nc',np.stack(b,-1),sh)+.5
    color_clamped=raw_colors.real<0
    colors=np.where(color_clamped,0,raw_colors)
    depths=np.c_[p.mean(1),np.ones(n)]@f['view'].astype(float).T
    order=depths[:,2].real.argsort().tolist()
    trans=np.ones(len(f['pixels']),dtype=theta.dtype)
    rgb=np.zeros((len(trans),3),dtype=theta.dtype)
    done=np.zeros(len(trans),dtype=bool)
    included=[];terminated=[]
    for i in order:
        stop=valid[i]&(np.real(trans*(1-alpha[i]))<1e-4)&~done
        take=valid[i]&~done&~stop
        a=np.where(take,alpha[i],0)
        rgb+=((trans*a)[:,None]*colors[i])
        trans*=1-a
        done|=stop
        included.append(take.tolist());terminated.append(stop.tolist())
    rgb+=trans[:,None]*f['background'].astype(float)
    fp=dict(order=order,active=active.tolist(),inside=inside.tolist(),valid=valid.tolist(),
            capped=cap.tolist(),color_clamped=color_clamped.tolist(),included=included,terminated=terminated)
    return (rgb*f['pixel_grad'].astype(float)).sum(),rgb,fp


def complex_gradient(theta,f):
    out=[]
    for j in range(len(theta)):
        x=theta.astype(complex)
        x[j]+=1e-20j
        out.append(evaluate(x,f)[0].imag/1e-20)
    return np.array(out)
