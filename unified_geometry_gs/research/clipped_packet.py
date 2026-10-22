"""Recover pinhole ray coefficients from the full affine packet, including center clip."""
from __future__ import annotations
import numpy as np

def recover(plane,width,conic,qmean,qbar,qray,focal,amplitude):
    z=plane[...,2]/np.sqrt(np.sum(qmean*qmean,axis=-1))
    rbar=np.sum(qbar*qbar,axis=-1)
    b=rbar/(width*width);scale=z*z*b
    linear=plane[...,:2]*focal
    c=(linear*np.sqrt(rbar)[...,None]/z[...,None]+qbar[...,:2])/rbar[...,None]
    kxx=conic[...,0]*focal[...,0]**2/scale
    kxy=conic[...,1]*focal[...,0]*focal[...,1]/scale
    kyy=conic[...,2]*focal[...,1]**2/scale
    a=qray[...,:2]-qbar[...,:2];m=qmean[...,:2]-qbar[...,:2]
    A=1+np.sum(c*a,axis=-1);B=1+np.sum(c*m,axis=-1)
    aa=kxx*a[...,0]**2+2*kxy*a[...,0]*a[...,1]+kyy*a[...,1]**2
    am=a[...,0]*(kxx*m[...,0]+kxy*m[...,1])+a[...,1]*(kxy*m[...,0]+kyy*m[...,1])
    denominator=A*A+aa;cross=A*B+am
    v=A[...,None]*m-B[...,None]*a
    detK=kxx*kyy-kxy*kxy
    wedge=a[...,0]*m[...,1]-a[...,1]*m[...,0]
    gram_det=detK*wedge*wedge+kxx*v[...,0]**2+2*kxy*v[...,0]*v[...,1]+kyy*v[...,1]**2
    energy=scale*gram_det/denominator
    raynorm=np.sqrt(np.sum(qray*qray,axis=-1))
    peak=z*raynorm*cross/denominator
    sigma=raynorm/np.sqrt(b*denominator)
    alpha=amplitude*np.exp(-energy/2)
    return peak,sigma,alpha,{'denominator':denominator,'detK':detK,'gram_det':gram_det,'energy':energy,
                            'c':c,'K':np.stack((kxx,kxy,kyy),axis=-1)}
