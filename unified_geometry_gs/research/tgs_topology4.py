"""Finite equal-count, equal-area subdivision catalogue and geometry scoring.

The bisections are classical controls. Training-view maximin is an experimental
selector, not an established novel or beneficial training algorithm.
"""
import numpy as np

EDGES=((0,1),(1,2),(2,0))


def bisect(v,edge):
    i,j=EDGES[edge];k=3-i-j;m=(v[i]+v[j])/2
    return np.array([[v[i],m,v[k]],[m,v[j],v[k]]],dtype=v.dtype)


def children(v,pattern):
    if pattern['kind']=='red4':
        a,b,c=v;ab,ac,bc=(a+b)/2,(a+c)/2,(b+c)/2
        return np.array([[a,ab,ac],[b,ab,bc],[c,ac,bc],[ab,ac,bc]],dtype=v.dtype)
    e0,e1,e2=pattern['edges'];first=bisect(v,e0)
    return np.concatenate([bisect(first[0],e1),bisect(first[1],e2)])


def catalogue():
    result=[{'kind':'red4'}]
    result.extend({'kind':'binary4','edges':[e0,e1,e2]}
                  for e0 in range(3) for e1 in range(3) for e2 in range(3))
    eye=np.eye(3,dtype=np.float64)
    for i,p in enumerate(result):
        p['id']=i
        bary=children(eye,p)
        assert np.all(bary>=0) and np.array_equal(bary.sum(-1),np.ones((4,3)))
        assert np.all(bary*4==np.rint(bary*4))
        # Exact dyadic arithmetic: standard reference triangle area ratios.
        xy=bary[:,:,1:]
        area=np.abs(np.linalg.det(np.stack([xy[:,1]-xy[:,0],xy[:,2]-xy[:,0]],axis=1)))
        assert np.allclose(area,.25,rtol=0,atol=1e-15)
        p['barycentric_quarters']=(bary*4).astype(int).tolist()
    return result


def longest_edge(v):
    lengths=[np.sum((v[i]-v[j])**2) for i,j in EDGES]
    return int(np.argmax(lengths))


def world_longest_pattern(v,patterns):
    e0=longest_edge(v.astype(float));c=bisect(v.astype(float),e0)
    es=[e0,longest_edge(c[0]),longest_edge(c[1])]
    return next(p for p in patterns if p.get('edges')==es)


def camera_matrices(cameras):
    out=[]
    for c in cameras:
        c2w=np.eye(4);c2w[:3,:3]=c['rotation'];c2w[:3,3]=c['position']
        view=np.linalg.inv(c2w)
        p=np.zeros((4,4));p[0,0]=2*c['fx']/c['width'];p[1,1]=2*c['fy']/c['height']
        p[2,2]=100/99.99;p[2,3]=-1/99.99;p[3,2]=1
        out.append((view,p@view))
    return np.array([x[0] for x in out]),np.array([x[1] for x in out])


def project(v,projection,w,h):
    hom=np.einsum('...j,vij->v...i',np.concatenate([v,np.ones((*v.shape[:-1],1))],axis=-1),projection)
    denom=hom[...,3:4]+1e-7
    safe=np.where(np.abs(denom)>1e-12,denom,1e-12)
    pixel=((hom[...,:2]/safe+1)*[w,h]-1)/2
    return pixel,hom[...,3]


def inradius(pixel):
    edge=np.roll(pixel,-1,axis=-2)-pixel
    e1=pixel[...,1,:]-pixel[...,0,:];e2=pixel[...,2,:]-pixel[...,0,:]
    twice=np.abs(e1[...,0]*e2[...,1]-e1[...,1]*e2[...,0])
    perimeter=np.linalg.norm(edge,axis=-1).sum(-1)
    return twice/np.maximum(perimeter,1e-30)


def score_catalogue(v,patterns,cameras,w,h):
    view,projection=camera_matrices(cameras)
    points,depth=project(v,projection,w,h)
    center=v.mean(0);center_pixel,_=project(center[None],projection,w,h)
    extent=np.linalg.norm(points-center_pixel[:,0,None,:],axis=-1).max(-1)
    vp=np.einsum('j,vij->vi',np.r_[center,1],view)[:,:3]
    normal=np.cross(v[1]-v[0],v[2]-v[0]);normal=np.einsum('j,vij->vi',normal,view[:,:3,:3])
    normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-4)
    cosine=np.abs(np.sum(normal*vp,axis=-1)/np.maximum(np.linalg.norm(vp,axis=-1),1e-4))
    valid=(depth.min(-1)>.2)&(inradius(points)>=1)&(extent>=1)&(extent<=1600)&(cosine>=.001)
    valid&=(points[:,:,0].max(-1)>=0)&(points[:,:,0].min(-1)<w)&(points[:,:,1].max(-1)>=0)&(points[:,:,1].min(-1)<h)
    if not valid.any():
        return np.zeros(len(patterns)),np.zeros((0,len(patterns),4)),valid
    ideal=np.array([children(v.astype(float),p) for p in patterns])
    pc,dc=project(ideal,projection[valid],w,h)
    radii=inradius(pc)
    assert np.isfinite(radii).all() and (dc>.2-1e-7).all()
    scores=radii.min(axis=(0,2))
    return scores,radii,valid
