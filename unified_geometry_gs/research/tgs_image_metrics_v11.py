"""Explicit PNG quantization and float64 image metrics; no hidden crops or fitting."""
import math
import numpy as np
from scipy.ndimage import correlate1d

def png8(raw_chw):
    assert raw_chw.dtype==np.float32 and np.isfinite(raw_chw).all()
    x=np.multiply(raw_chw,np.float32(255),dtype=np.float32);x=np.add(x,np.float32(.5),dtype=np.float32);return np.ascontiguousarray(np.clip(x,0,255).astype(np.uint8).transpose(1,2,0))
def coordinates(h,w):
    ys=sorted({0,1,5,h//4,h//2,3*h//4,h-6,h-2,h-1});xs=sorted({0,1,5,w//4,w//2,3*w//4,w-6,w-2,w-1});return [[x,y] for y in ys for x in xs if 0<=x<w and 0<=y<h]
def gaussian():
    x=np.arange(-5,6,dtype=np.float64);g=np.exp(-x*x/(2*1.5**2));return g/g.sum()
def blur(a,g):return correlate1d(correlate1d(a,g,axis=0,mode='constant',cval=0.),g,axis=1,mode='constant',cval=0.)
def ssim_map(pred,gt):
    g=gaussian();mx=blur(pred,g);my=blur(gt,g);vx=blur(pred*pred,g)-mx*mx;vy=blur(gt*gt,g)-my*my;cov=blur(pred*gt,g)-mx*my;return ((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
def score(pred,gt):
    assert pred.shape==gt.shape and pred.dtype==gt.dtype==np.float64 and np.isfinite(pred).all() and np.isfinite(gt).all();mse=float(np.mean((pred-gt)**2));ss=ssim_map(pred,gt);h,w=gt.shape[:2];points=coordinates(h,w)
    return dict(MSE=mse,PSNR_dB=-10*math.log10(mse) if mse else float('inf'),SSIM=float(ss.mean()),SSIM_by_channel=ss.mean((0,1)).tolist(),SSIM_samples=[dict(pixel=[x,y],values=ss[y,x].tolist()) for x,y in points],pixels=h*w,SSIM_boundary='same-size zero padding5, Gaussian11 sigma1.5; RGB mean; no crop')
