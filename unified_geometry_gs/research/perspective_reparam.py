"""Exact-real pinhole ray coefficients from an unclipped, unfiltered affine packet."""
from __future__ import annotations
import numpy as np

def recover(t0,s0,linear_peak_shift,energy,q0,delta,amplitude):
    r=np.sum(q0*q0,axis=-1)
    g=np.sum(q0*delta,axis=-1)/r
    k=np.sum(delta*delta,axis=-1)/r
    length_ratio=1+2*g+k
    ell=g-linear_peak_shift/t0
    chi=energy*(s0/t0)**2
    denominator=(1+ell)**2+chi
    scale=np.sqrt(length_ratio)
    peak=t0*scale*(1+ell)/denominator
    width=s0*scale/np.sqrt(denominator)
    alpha=amplitude*np.exp(-energy/(2*denominator))
    return peak,width,alpha,denominator
