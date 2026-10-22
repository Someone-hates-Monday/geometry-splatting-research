"""Equivalent transverse-energy formula preserves on-axis zero derivatives."""
import numpy as np
import torch
from research.history_gap_gpu import Engine as BaseEngine,rotation,inverse3,_Gap
class Engine(BaseEngine):
    def __init__(self,raw,source,sections,metadata,native,device='cuda'):
        self.zero_kernel=bool(np.all(np.asarray(metadata)[:,4]==0))
        super().__init__(raw,source,sections,metadata,native,device)
    def screen(self,mu,R,scales):
        if self.zero_kernel:return torch.zeros_like(mu[:,0])
        gamma=super().screen(mu,R,scales)
        return torch.where(self.meta[self.ids,4]==0,torch.zeros_like(gamma),gamma)
    def fields(self,theta):
        base=self.base;src=self.source;variance=theta[:,3:6].exp().square();v0=base[:,3:6].exp().square();f=variance+self.radius[:,None]**2;f0=v0+self.radius[:,None]**2
        scales=src[:,3:6]*torch.sqrt(f/f0);q=src[:,6:10]+(torch.nn.functional.normalize(theta[:,6:10],dim=1)-torch.nn.functional.normalize(base[:,6:10],dim=1));R=rotation(q)
        factor=torch.diag_embed(scales.reciprocal())@inverse3(R);direction=self.direction[self.ids];diff=theta[:,:3]-self.origin[self.ids];u=(factor@direction[:,:,None]).squeeze(-1);v=(factor@diff[:,:,None]).squeeze(-1);den=u.square().sum(1);peak=(u*v).sum(1)/den
        # ||Fv x Fd||²/||Fd||² = det(F)²||F^-T(v x d)||²/||Fd||².
        # F=S^-1 R^-1, hence F^-T=S R^T. No projected-residual subtraction.
        cross=torch.linalg.cross(diff,direction,dim=1);dual=(R.transpose(1,2)@cross[:,:,None]).squeeze(-1)*scales
        a,b,c=R.unbind(-1);detR=(a*torch.linalg.cross(b,c,dim=1)).sum(1)
        E=dual.square().sum(1)/((scales.prod(1)*detR).square()*den);rho=den.sqrt()
        gamma=self.screen(theta[:,:3],R,scales);logfilter=.5*(variance.log()-f.log()).sum(1);logfilter0=.5*(v0.log()-f0.log()).sum(1);amplitude=src[:,35]*torch.exp(torch.nn.functional.logsigmoid(theta[:,10])-torch.nn.functional.logsigmoid(base[:,10])+logfilter-logfilter0+gamma-self.gamma0)
        return peak.contiguous(),rho.contiguous(),E.contiguous(),amplitude.contiguous()
