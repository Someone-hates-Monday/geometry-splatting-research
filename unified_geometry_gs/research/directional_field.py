"""Pointwise single-normal attenuation diagnostics; no renderer dependencies."""
from __future__ import annotations
import math
import mpmath as mp
import numpy as np


def orthogonal_witness(candidate, gram):
    norm=np.linalg.norm(candidate)
    if norm==0:
        return np.linalg.eigh(gram)[1][:,-1]
    u=candidate/norm
    seed=np.eye(3)[int(np.argmin(abs(u)))]
    v=np.cross(u,seed);v/=np.linalg.norm(v)
    Q=np.column_stack((v,np.cross(u,v)))
    return Q@np.linalg.eigh(Q.T@gram@Q)[1][:,-1]


def interval_gaussian_gradient(raw,point):
    """Enclose ideal gradients from exact binary raw parameters with directed rounding."""
    iv=mp.iv
    xyz=[iv.mpf(float(x)) for x in point]
    mu=[iv.mpf(float(x)) for x in raw['mean']]
    logscale=[iv.mpf(float(x)) for x in raw['log_scale']]
    filt=iv.mpf(float(raw['filter_3D']))
    scales=[iv.exp(2*x) for x in logscale]
    smooth=[x+filt*filt for x in scales]
    ratio=iv.mpf(1)
    for a,b in zip(scales,smooth):ratio*=a/b
    amp=iv.sqrt(ratio)/(1+iv.exp(-iv.mpf(float(raw['opacity_logit']))))
    q=[iv.mpf(float(x)) for x in raw['raw_quaternion']]
    norm=iv.sqrt(sum(x*x for x in q));w,x,y,z=[x/norm for x in q]
    R=[[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
       [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
       [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]
    delta=[x-y for x,y in zip(xyz,mu)]
    local=[sum(R[i][j]*delta[i] for i in range(3)) for j in range(3)]
    Q=sum(x*x/s for x,s in zip(local,smooth))
    G=amp*iv.exp(-Q/2)
    Pd=[sum(R[i][j]*local[j]/smooth[j] for j in range(3)) for i in range(3)]
    return [G*p/(2*(1-G)) for p in Pd]


def exact_endpoint(value):
    """Represent a binary mpmath endpoint as an exact rational, not a rounded string."""
    sign,man,exponent,_=mp.mpf(value)._mpf_
    numerator=(-1 if sign else 1)*man
    if exponent>=0:return {'numerator':str(numerator<<exponent),'denominator':'1'}
    return {'numerator':str(numerator),'denominator':str(1<<(-exponent))}


def interval_pair_certificate(raw_pair,point,dps=80):
    mp.iv.dps=dps;mp.mp.dps=dps
    a,b=[interval_gaussian_gradient(raw,point) for raw in raw_pair]
    cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
    det=sum(x*x for x in cross)
    trace=sum(x*x for x in a+b)
    positive=mp.mpf(det.a)>0 and mp.mpf(trace.a)>0
    cert=mp.iv.sqrt(det/trace) if positive else mp.iv.mpf(0)
    lower=mp.mpf(cert.a)
    return {'interval_dps':dps,'positive_certified':bool(positive),
            'gradient_intervals':[[[mp.nstr(mp.mpf(v.a),70),mp.nstr(mp.mpf(v.b),70)] for v in row] for row in (a,b)],
            'cross_squared_lower_exact':exact_endpoint(det.a),'trace_upper_exact':exact_endpoint(trace.b),
            'certificate_lower_exact':exact_endpoint(cert.a),
            'conservative_double_lower_bound':math.nextafter(float(lower),-math.inf) if positive else 0.,
            'meaning':'Lower bound on best uniform absolute-direction approximation error for the full ideal model, because PSD additions cannot decrease its second eigenvalue. Exact rational endpoints, not formatted gradient intervals, carry the certificate.'}
