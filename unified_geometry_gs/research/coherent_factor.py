"""CPU coherent inverse-factor construction and compact upper-triangular cache."""
from __future__ import annotations
import numpy as np

TRI=np.triu_indices(3)
VARIANTS=('qr32','qr_mixed','qr64')


def rotation(q):
    w,x,y,z=np.asarray(q,dtype=np.float64)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def inverse_quaternion_rotation(q):
    q=np.asarray(q,dtype=np.float64);v=q[1:]
    k=np.dot(q,q)-1.;den=1.+4.*k*np.dot(v,v)
    if den<=0:raise ValueError('Singular/nonpositive quaternion rotation determinant')
    return (rotation(q).T+4.*k*np.outer(v,v))/den


def construct(mean,scales,R,view,quaternion=None):
    view=np.asarray(view,dtype=np.float64);A=view[:3,:3].T
    inverse_R=np.linalg.inv(R) if quaternion is None else inverse_quaternion_rotation(quaternion)
    F=np.diag(1/np.asarray(scales,dtype=np.float64))@inverse_R@np.linalg.inv(A)
    camera_mean=A@np.asarray(mean,dtype=np.float64)+view[3,:3]
    _,U=np.linalg.qr(F)
    signs=np.where(np.diag(U)<0.,-1.,1.)
    U=signs[:,None]*U
    return U,camera_mean,F


def store(U,mean,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    upper=np.asarray(U[TRI],dtype=np.float64 if variant=='qr64' else np.float32)
    mu=np.asarray(mean,dtype=np.float32 if variant=='qr32' else np.float64)
    return upper,mu


def decode(upper,mean,qray,amplitude):
    U=np.zeros((3,3),dtype=np.float64);U[TRI]=np.asarray(upper,dtype=np.float64)
    q=np.asarray(qray,dtype=np.float64);d=q/np.linalg.norm(q)
    u=U@d;v=U@np.asarray(mean,dtype=np.float64)
    a=np.dot(u,u);peak=np.dot(u,v)/a
    cross=np.cross(v,u);energy=np.dot(cross,cross)/a
    rho=np.sqrt(a);alpha=min(float(np.float32(.99)),float(amplitude)*np.exp(-energy/2))
    valid=a>0 and energy>=0 and np.isfinite([peak,rho,alpha,energy]).all()
    return np.array([peak,1/rho,alpha]),rho,energy,bool(valid)
