"""Exact finite-median loss with a C1 continuation to insufficient mass."""
import mpmath as mp
import numpy as np
import torch
from research.history_gap_math import mp_root,mp_loss_gradient
from research.history_gap_raw import coefficients,torch_fields
from research.exact_outlier_world_reference import rotation,dot

def mp_extended(nodes,initial=None):
    omega=-mp.fsum(mp.log1p(-a) for a,p,rho in nodes)
    if omega<=mp.log(2):
        D=mp.log(4)-omega;g=[[-2*D/(1-a),mp.mpf(0),mp.mpf(0)] for a,p,rho in nodes]
        return D*D,g,dict(status='mass_extension',omega=omega,root=None,D=D)
    t=mp_root(nodes,initial);loss,g,info=mp_loss_gradient(nodes,t)
    return loss,g,dict(status='finite_root',omega=omega,root=t,D=info['D'])

def mp_path_nodes(raw,source,direction,h):
    raw=[[mp.mpf(float(v)) for v in row] for row in raw];src=[[mp.mpf(float(v)) for v in row] for row in source]
    s=src[0];A=mp.matrix([s[20+3*i:23+3*i] for i in range(3)]);Ai=A**-1;q=mp.matrix(s[32:35]);origin=-Ai*mp.matrix(s[29:32]);ray=Ai*q/mp.sqrt(dot(q,q));cap=mp.mpf(float(np.float32(.99)));nodes=[]
    for base,capture,d in zip(raw,src,direction):
        theta=[v+h*mp.mpf(float(delta)) for v,delta in zip(base,d)];f2=base[11]**2;v0=[mp.exp(2*v) for v in base[3:6]];v=[mp.exp(2*a) for a in theta[3:6]];scales=[capture[3+j]*mp.sqrt((v[j]+f2)/(v0[j]+f2)) for j in range(3)]
        q0=base[6:10];q1=theta[6:10];n0=mp.sqrt(mp.fsum(a*a for a in q0));n1=mp.sqrt(mp.fsum(a*a for a in q1));quat=[capture[6+j]+q1[j]/n1-q0[j]/n0 for j in range(4)];rot=rotation(quat);P=(rot*mp.diag([a*a for a in scales])*rot.T)**-1;diff=mp.matrix(theta[:3])-origin;den=dot(ray,P*ray);p=dot(ray,P*diff)/den;w=diff-p*ray;E=dot(w,P*w)
        amp=capture[35]*(1+mp.exp(-base[10]))/(1+mp.exp(-theta[10]))*mp.sqrt(mp.fprod(v[j]/(v[j]+f2)*(v0[j]+f2)/v0[j] for j in range(3)));nodes.append([min(cap,amp*mp.exp(-E/2)),p,mp.sqrt(den)])
    return nodes

def torch_absent_gradient(theta_value,raw,source,meta):
    theta=torch.tensor(theta_value,dtype=torch.float64,requires_grad=True);fields=torch_fields(theta,raw,source,meta,'factor');p,rho,la=coefficients(fields);omega=-torch.log1p(-la.exp()).sum()
    if float(omega.detach())>np.log(2):raise ValueError('This diagnostic branch is only for insufficient mass')
    loss=(np.log(4)-omega)**2;g=torch.autograd.grad(loss,theta)[0]
    return float(loss.detach()),g.detach().numpy(),float(omega.detach())
