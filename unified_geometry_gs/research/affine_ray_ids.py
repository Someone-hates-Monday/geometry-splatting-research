"""Read-only affine CPU copy carrying original PLY IDs; parity is checked at runtime."""
from __future__ import annotations
import numpy as np


def project_with_ids(model,camera,resolution=2,kernel=0.):
    xyz,scales2,opacity,rotation=model
    width=round(camera['width']/resolution);height=round(camera['height']/resolution)
    fx=camera['fx']*width/camera['width'];fy=camera['fy']*height/camera['height']
    c2w=np.asarray(camera['rotation']);xyz_cam=(xyz-camera['position'])@c2w
    keep=xyz_cam[:,2]>.2;near_ids=np.flatnonzero(keep)
    xyz_cam,scales2,opacity,rotation=xyz_cam[keep],scales2[keep],opacity[keep],rotation[keep]
    tc=np.linalg.norm(xyz_cam,axis=1);z=xyz_cam[:,2]
    b=np.einsum('ij,njk->nik',c2w.T,rotation)
    cov=np.einsum('nik,nk,njk->nij',b,scales2,b)
    precision=np.einsum('nik,nk,njk->nij',b,1/scales2,b)
    u=np.clip(xyz_cam[:,0]/z,-1.3*width/(2*fx),1.3*width/(2*fx))
    v=np.clip(xyz_cam[:,1]/z,-1.3*height/(2*fy),1.3*height/(2*fy))
    jx=np.column_stack((fx/z,np.zeros_like(z),-fx*u/z))
    jy=np.column_stack((np.zeros_like(z),fy/z,-fy*v/z))
    covxx=np.einsum('ni,nij,nj->n',jx,cov,jx)
    covxy=np.einsum('ni,nij,nj->n',jx,cov,jy)
    covyy=np.einsum('ni,nij,nj->n',jy,cov,jy)
    det0=np.maximum(1e-6,covxx*covyy-covxy**2)
    covxx,covyy=covxx+kernel,covyy+kernel
    det=covxx*covyy-covxy**2
    opacity=opacity*np.sqrt(det0/np.maximum(1e-6,det))
    conic=np.column_stack((covyy/det,-covxy/det,covxx/det))
    mid=.5*(covxx+covyy)
    radius=np.ceil(3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid**2-det))))
    uv=np.column_stack((fx*xyz_cam[:,0]/z+(width-1)/2,fy*xyz_cam[:,1]/z+(height-1)/2))
    grid=np.array([(width+15)//16,(height+15)//16])
    tile_min=np.minimum(grid,np.maximum(0,np.trunc((uv-radius[:,None])/16))).astype(int)
    tile_max=np.minimum(grid,np.maximum(0,np.trunc((uv+radius[:,None]+15)/16))).astype(int)
    uvh=np.column_stack((u,v,np.ones_like(z)))
    qm=np.einsum('nij,nj->ni',precision,uvh)
    vb=np.sum(qm*uvh,axis=1);ray_len2=1+u*u+v*v;factor=z/np.sqrt(ray_len2)
    px=((v*v+1)*qm[:,0]-u*v*qm[:,1]-u*qm[:,2])/vb
    py=(-u*v*qm[:,0]+(u*u+1)*qm[:,1]-v*qm[:,2])/vb
    plane=np.column_stack((px*factor/fx,py*factor/fy,tc));widths=np.sqrt(ray_len2/vb)
    keep=(det>0)&np.all(tile_max>tile_min,axis=1)
    order=np.flatnonzero(keep)[np.argsort(tc[keep],kind='stable')]
    return {'uv':uv[order],'conic':conic[order],'opacity':opacity[order],
            'plane':plane[order],'widths':widths[order],'tile_min':tile_min[order],'tile_max':tile_max[order],
            'image_width':width,'image_height':height,'fx':fx,'fy':fy,
            '_near_ids':near_ids,'_gaussian_ids':near_ids[order]}


def ray_with_ids(projected,pixel):
    tile=np.asarray(pixel)//16
    keep=np.all((projected['tile_min']<=tile)&(tile<projected['tile_max']),axis=1)
    tile_ids=np.flatnonzero(keep);ids=tile_ids
    diff=projected['uv'][ids]-pixel;c=projected['conic'][ids]
    power=-.5*(c[:,0]*diff[:,0]**2+c[:,2]*diff[:,1]**2)-c[:,1]*diff[:,0]*diff[:,1]
    raw_alpha=projected['opacity'][ids]*np.exp(np.minimum(power,0))
    alpha=np.minimum(.99,raw_alpha);keep=(power<=0)&(alpha>=1/255)
    ids,diff,alpha=ids[keep],diff[keep],alpha[keep]
    eligible_ids=ids.copy();eligible_capped_count=int(sum(raw_alpha[keep]>.99))
    prefix=np.cumprod(1-alpha);stop=int(np.searchsorted(-prefix,-1e-4,side='right'))
    ids,diff,alpha=ids[:stop],diff[:stop],alpha[:stop]
    plane=projected['plane'][ids]
    peaks=plane[:,2]+(plane[:,:2]*diff).sum(axis=1);widths=projected['widths'][ids]
    init=0.
    if len(alpha):
        trans_before=np.r_[1.,np.cumprod(1-alpha)[:-1]]
        init=float(peaks[np.flatnonzero(trans_before>.5)[-1]])
    original=projected['_gaussian_ids']
    return {'alpha':alpha,'peaks':peaks,'widths':widths,'init':init,
            'memberships':{'selected':original[ids],'eligible':original[eligible_ids],
                           'tile':original[tile_ids],'valid':original,'near':projected['_near_ids']},
            'eligible_capped_count':eligible_capped_count,
            'tile_positive_power_count':int(sum(power>0)),
            'tile_below_alpha_threshold_count':int(sum(alpha_ < 1/255 for alpha_ in np.minimum(.99,raw_alpha)))}


def verify_projection(reference,carried):
    for key,value in reference.items():
        if isinstance(value,np.ndarray):assert np.array_equal(value,carried[key]),key
        else:assert value==carried[key],key
