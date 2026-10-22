"""Bounded adjoint of squared history gap in coefficient coordinates."""
import math
import numpy as np
import mpmath as mp

def mp_residual(nodes,t):
    beta=mp.mpf(1);tails=[]
    for a,p,rho in nodes:
        x=(t-p)*rho;q=-mp.log1p(-a*mp.exp(-x*x/2))/2
        if t>p:beta*=1-a;tails.append(-q)
        else:tails.append(q)
    return mp.log(mp.mpf('.5')/beta)+mp.fsum(tails)

def mp_slope(nodes,t):return mp.fsum(a*mp.exp(-((t-p)*rho)**2/2)*abs((t-p)*rho)*rho/(2*(1-a*mp.exp(-((t-p)*rho)**2/2))) for a,p,rho in nodes)

def mp_root(nodes,initial=None):
    if mp.fprod(1-a for a,p,rho in nodes)>=mp.mpf('.5'):raise ValueError('No finite root')
    if len(nodes)==1:
        a,p,rho=nodes[0]
        if a==mp.mpf('.75'):return p
        if a>mp.mpf('.75'):return p-mp.sqrt(-2*mp.log(mp.mpf('.75')/a))/rho
        return p+mp.sqrt(-2*mp.log((1-4*(1-a)**2)/a))/rho
    if len(nodes)==2 and all(a==mp.mpf('.5') for a,p,rho in nodes):return (nodes[0][1]*nodes[0][2]+nodes[1][1]*nodes[1][2])/(nodes[0][2]+nodes[1][2])
    tolerance=mp.power(10,-mp.mp.dps+45)
    if initial is not None:
        t=mp.mpf(initial)
        for _ in range(25):
            value=mp_residual(nodes,t);s=mp_slope(nodes,t)
            if s<=0:break
            change=value/s
            if abs(change)>1:break
            t-=change
            if abs(change)<tolerance:return t
    reach=max(mp.mpf(1),10/min(r for a,p,r in nodes));lo=min(p for a,p,r in nodes)-reach;hi=max(p for a,p,r in nodes)+reach
    assert mp_residual(nodes,lo)<0<mp_residual(nodes,hi)
    for _ in range(2000):
        mid=(lo+hi)/2;v=mp_residual(nodes,mid)
        if v<0:lo=mid
        elif v>0:hi=mid
        else:return mid  # Numeric candidate, checked by independent identities/FD.
        if hi-lo<tolerance:return (lo+hi)/2
    raise ValueError('MP bisection unresolved')

def mp_loss_gradient(nodes,t):
    history=[];ell=[];densities=[];parts=[]
    for a,p,rho in nodes:
        x=(t-p)*rho;g=mp.exp(-x*x/2);u=a*g;b=u/(2*(1-u));after=t>p
        ell.append(mp.log1p(-u)/2);history.append(mp.log1p(a/(1-a)*(-mp.expm1(-x*x/2))) if after else mp.mpf(0));densities.append(b*abs(x)*rho);parts.append((a,x,rho,b,after))
    D=mp.fsum(history);den=mp.fsum(densities)
    if den==0:
        if D!=0:raise ValueError('Zero density with nonzero loss')
        return D*D,[[mp.mpf(0)]*3 for _ in nodes],dict(D=D,r=mp.mpf(0),Ht=den,ell=mp.fsum(ell),singular=True)
    mass_after=mp.fsum(v for v,part in zip(densities,parts) if part[4]);mass_before=mp.fsum(v for v,part in zip(densities,parts) if not part[4]);ratio=(mass_after-mass_before)/den;grad=[]
    for a,x,rho,b,after in parts:
        factor=2*(mass_before if after else mass_after)/den;c=b*factor;grad.append([2*D*(-factor*g_at(x)/(2*(1-a*g_at(x)))-ratio/(1-a)*int(after)),2*D*(-c*x*rho),2*D*c*x*x/rho])
    return D*D,grad,dict(D=D,r=ratio,Ht=den,ell=mp.fsum(ell),singular=False)

def g_at(x):return mp.exp(-x*x/2)

def float_loss_gradient(coeff,t,log_alpha=None):
    a,p,rho=coeff.T;x=(t-p)*rho;after=t>p
    if log_alpha is None:
        log_alpha=np.full_like(a,-np.inf);positive=a>0;log_alpha[positive]=np.log(a[positive])
    lg=log_alpha-.5*x*x;u=np.exp(lg);b=u/(2*(1-u));terms=np.zeros_like(a);terms[after]=np.log1p(a[after]/(1-a[after])*(-np.expm1(-.5*x[after]**2)));D=float(terms.sum());logdensity=np.full_like(a,-np.inf);nz=x!=0;logdensity[nz]=lg[nz]+np.log(abs(x[nz]))+np.log(rho[nz])-math.log(2)-np.log1p(-u[nz]);scale=float(np.max(logdensity))
    if not np.isfinite(scale):
        if D!=0:raise FloatingPointError('density unavailable')
        return 0.,np.zeros_like(coeff),dict(D=0.,r=0.,singular=True)
    weights=np.exp(logdensity-scale);mass_after=float(np.sum(weights[after]));mass_before=float(np.sum(weights[~after]));den=mass_after+mass_before;ratio=(mass_after-mass_before)/den;factor=2*np.where(after,mass_before,mass_after)/den;c=b*factor
    # b/alpha = g/[2(1-alpha*g)] also at a rounded zero alpha.
    # Preserve log(alpha) for relative density weights; alpha may underflow.
    gradient=2*D*np.stack([-factor*np.exp(-.5*x*x)/(2*(1-u))-ratio*after/(1-a),-c*x*rho,c*x*x/rho],axis=1)
    return D*D,gradient,dict(D=D,r=ratio,singular=False)
