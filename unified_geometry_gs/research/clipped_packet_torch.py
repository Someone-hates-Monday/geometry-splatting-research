"""Torch reference graphs for raw-parameter chain checking, not a production backend."""
from __future__ import annotations
import torch

def world_parameters(raw):
    variance=torch.exp(2*raw[:,3:6]);smooth=variance+raw[:,11,None]**2
    q=raw[:,6:10]/torch.linalg.vector_norm(raw[:,6:10],dim=1,keepdim=True)
    w,x,y,z=q.unbind(1)
    R=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                   2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                   2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=1).reshape(-1,3,3)
    amplitude=torch.sigmoid(raw[:,10])*torch.sqrt(torch.prod(variance/smooth,dim=1))
    return smooth,R,amplitude

def recover(plane,width,conic,qmean,qbar,qray,focal,amplitude):
    z=plane[:,2]/torch.linalg.vector_norm(qmean,dim=1)
    rbar=torch.sum(qbar*qbar,dim=1);b=rbar/(width*width);scale=z*z*b
    c=(plane[:,:2]*focal*torch.sqrt(rbar)[:,None]/z[:,None]+qbar[:,:2])/rbar[:,None]
    kxx=conic[:,0]*focal[0]**2/scale;kxy=conic[:,1]*focal[0]*focal[1]/scale;kyy=conic[:,2]*focal[1]**2/scale
    a=qray[:,:2]-qbar[:,:2];m=qmean[:,:2]-qbar[:,:2]
    A=1+torch.sum(c*a,dim=1);B=1+torch.sum(c*m,dim=1)
    aa=kxx*a[:,0]**2+2*kxy*a[:,0]*a[:,1]+kyy*a[:,1]**2
    am=a[:,0]*(kxx*m[:,0]+kxy*m[:,1])+a[:,1]*(kxy*m[:,0]+kyy*m[:,1])
    den=A*A+aa;cross=A*B+am;v=A[:,None]*m-B[:,None]*a
    det=kxx*kyy-kxy*kxy;wedge=a[:,0]*m[:,1]-a[:,1]*m[:,0]
    gdet=det*wedge*wedge+kxx*v[:,0]**2+2*kxy*v[:,0]*v[:,1]+kyy*v[:,1]**2
    energy=scale*gdet/den;raynorm=torch.linalg.vector_norm(qray,dim=1)
    return torch.stack((z*raynorm*cross/den,raynorm/torch.sqrt(b*den),amplitude*torch.exp(-energy/2)),dim=1)

def packet_from_raw(raw,origin,c2w,focal,center,limit,qray):
    smooth,rotation,amplitude=world_parameters(raw)
    cam=(raw[:,:3]-origin)@c2w;z=cam[:,2];t0=torch.linalg.vector_norm(cam,dim=1)
    qmean=torch.cat((cam[:,:2]/z[:,None],torch.ones_like(z[:,None])),dim=1)
    qbar=torch.cat((torch.clamp(qmean[:,:2],min=-limit,max=limit),torch.ones_like(z[:,None])),dim=1)
    axes=c2w.T[None]@rotation
    cov=torch.einsum('nik,nk,njk->nij',axes,smooth,axes)
    precision=torch.einsum('nik,nk,njk->nij',axes,1/smooth,axes)
    pq=torch.einsum('nij,nj->ni',precision,qbar);b=torch.sum(pq*qbar,dim=1);rbar=torch.sum(qbar*qbar,dim=1)
    width=torch.sqrt(rbar/b)
    linear=z[:,None]/torch.sqrt(rbar)[:,None]*(rbar[:,None]*pq/b[:,None]-qbar)[:,:2]/focal
    plane=torch.cat((linear,t0[:,None]),dim=1)
    zero=torch.zeros_like(z)
    J=torch.stack((focal[0]/z,zero,-focal[0]*qbar[:,0]/z,
                   zero,focal[1]/z,-focal[1]*qbar[:,1]/z),dim=1).reshape(-1,2,3)
    screen=J@cov@J.transpose(1,2);inverse=torch.linalg.inv(screen)
    conic=torch.stack((inverse[:,0,0],inverse[:,0,1],inverse[:,1,1]),dim=1)
    uv=qmean[:,:2]*focal+center
    reconstructed=torch.cat(((uv-center)/focal,torch.ones_like(z[:,None])),dim=1)
    return recover(plane,width,conic,reconstructed,qbar,qray,focal,amplitude),qmean

def direct_world(raw,origin,direction):
    smooth,rotation,amplitude=world_parameters(raw)
    local_o=torch.einsum('nij,ni->nj',rotation,origin-raw[:,:3])
    local_d=torch.einsum('nij,j->ni',rotation.transpose(1,2),direction)
    quadratic=torch.sum(local_d*local_d/smooth,dim=1)
    peak=-torch.sum(local_o*local_d/smooth,dim=1)/quadratic
    perpendicular=local_o+peak[:,None]*local_d
    energy=torch.sum(perpendicular*perpendicular/smooth,dim=1)
    return torch.stack((peak,1/torch.sqrt(quadratic),amplitude*torch.exp(-energy/2)),dim=1)
