"""Research-only squared-gap raw VJP and source-preserving Torch graph."""
import math
import numpy as np
import mpmath as mp
import torch
from research.history_gap_math import mp_loss_gradient,float_loss_gradient
from research.world_root_vjp import mp_rotation_derivatives,mp_screen_derivatives,inner
from research.world_root_vjp_v2 import torch_fields,polish

def mp_raw_gradient(geometry,source,raw,meta,t):
    from research.exact_outlier_world_reference import dot,vector
    nodes=[[g['alpha'],g['p'],g['rho']] for g in geometry]
    loss,coefficient,info=mp_loss_gradient(nodes,t);result=[];caps=0
    cap=mp.mpf(float(np.float32(.99)))
    for g,row,r,seeds in zip(geometry,source,raw,coefficient):
        ha=seeds[0]*g['alpha'];hp=seeds[1];hr=seeds[2]
        uncapped=g['amp']*mp.exp(-g['E']/2)
        if uncapped==cap:raise ValueError('exact effective-alpha cap boundary')
        if uncapped>cap:ha=mp.mpf(0);caps+=1
        he=-ha/2;P=g['P'];d=g['d'];w=g['w'];a=g['a']
        mu=hp*(P*d)/a+2*he*(P*w)
        HP=hp*(d*w.T+w*d.T)/(2*a)+hr*(d*d.T)/(2*g['rho'])+he*(w*w.T)
        Ri=g['R']**-1;F=g['S']**-1*Ri
        ss=[-2*dot(F[j,:].T,HP*F[j,:].T) for j in range(3)]
        GR=-2*P*HP*Ri.T;qq=[inner(GR,v) for v in mp_rotation_derivatives([mp.mpf(float(x)) for x in row[6:10]])]
        if meta['kernel']==0:gm=mp.zeros(3,1);gs=[mp.mpf(0)]*3;gq=[mp.mpf(0)]*4
        else:gm,gs,gq,_=mp_screen_derivatives(g,row,meta)
        mu+=ha*gm;radius=mp.mpf(float(r[11]))
        kv=[mp.exp(2*mp.mpf(float(r[3+j])))/(mp.exp(2*mp.mpf(float(r[3+j])))+radius*radius) for j in range(3)]
        scale=[ss[j]*kv[j]+ha*(1-kv[j]+gs[j]*kv[j]) for j in range(3)]
        qraw=vector(r[6:10]);norm=mp.sqrt(dot(qraw,qraw));J=(mp.eye(4)-qraw*qraw.T/(norm*norm))/norm
        quat=J.T*mp.matrix([qq[j]+ha*gq[j] for j in range(4)])
        eta=ha/(1+mp.exp(mp.mpf(float(r[10]))));result.append(list(mu)+scale+list(quat)+[eta])
    return loss,np.array([[float(v) for v in row] for row in result]),dict(D=float(info['D']),ratio=float(info['r']),capped_nodes=caps)

def coefficients(fields):
    p,rho,E,A=fields[:4]
    if not bool((A>0).all()):raise FloatingPointError('nonpositive captured amplitude')
    la=(A.log()-E/2).clamp_max(math.log(float(np.float32(.99))))
    return p,rho,la

def own_root(fields):
    p,rho,la=coefficients(fields)
    omega=float((-torch.log1p(-la.detach().exp())).sum())
    if omega<=math.log(2):return None,'float_insufficient_mass',0
    reach=max(1.,10/float(rho.detach().min()));low=float(p.detach().min());high=float(p.detach().max())
    for expansion in range(16):
        t,status,steps=polish(fields,low-reach,high+reach)
        if t is not None:return t,status,steps
        reach*=2
    return None,'unbracketed_after_expansion',0

def torch_gap_gradient(raw,source,meta,arm):
    theta=torch.tensor(raw[:,:11],dtype=torch.float64,requires_grad=True)
    fields=torch_fields(theta,raw,source,meta,arm);t,status,steps=own_root(fields)
    if t is None:raise FloatingPointError(status)
    p,rho,la=coefficients(fields)
    c=np.stack([la.detach().exp().numpy(),p.detach().numpy(),rho.detach().numpy()],axis=1)
    loss,g,info=float_loss_gradient(c,t,la.detach().numpy())
    # Implicit coefficient adjoint already includes root motion. Contract only
    # these first-order seeds; differentiating the seeds again is not needed.
    seeds=torch.tensor(g,dtype=torch.float64)
    scalar=(p*seeds[:,1]+rho*seeds[:,2]+la*(seeds[:,0]*la.detach().exp())).sum()
    gradient=torch.autograd.grad(scalar,theta)[0].detach().numpy()
    return loss,gradient,dict(root=t,root_status=status,root_steps=steps,**info)

def torch_forward(theta,raw,source,meta,arm='factor'):
    with torch.no_grad():
        fields=torch_fields(torch.tensor(theta,dtype=torch.float64),raw,source,meta,arm)
        t,status,steps=own_root(fields)
        if t is None:return None,dict(root_status=status)
        p,rho,la=coefficients(fields);c=np.stack([la.exp().numpy(),p.numpy(),rho.numpy()],axis=1)
        loss,_,info=float_loss_gradient(c,t,la.numpy())
        return loss,dict(root=t,root_status=status,D=info['D'])

def parameter_scale(raw,source):
    scale=np.ones_like(raw[:,:11]);scale[:,:3]=np.min(source[:,3:6],axis=1)[:,None]
    scale[:,6:10]=np.linalg.norm(raw[:,6:10],axis=1)[:,None]
    return scale
