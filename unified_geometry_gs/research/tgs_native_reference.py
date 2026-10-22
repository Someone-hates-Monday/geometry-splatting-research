"""Independent NumPy point sampler for the fixed TGS forward diagnostics.

Implements pinned source's finite tile rectangle, not an ideal pixel integral.
Only constant colors; SH is deliberately outside this reference.
"""
import numpy as np


def projected(points, projection, width, height):
    hom = np.concatenate([points,np.ones((*points.shape[:-1],1))],axis=-1)@projection.T
    ndc = hom[...,:2]/(hom[...,3:4]+1e-7)
    return ((ndc+1)*[width,height]-1)/2


def render(points, sigma, opacity, view, projection, width=256, height=256):
    points = np.asarray(points,dtype=np.float64)
    sigma, opacity = np.asarray(sigma,dtype=float), np.asarray(opacity,dtype=float)
    yy,xx = np.mgrid[:height,:width]
    pixel = np.stack([xx,yy],axis=-1)
    images, radii, records = [],[],[]
    for v,sig,o in zip(points,sigma,opacity):
        center = v.mean(0)
        vc = (view@np.r_[center,1])[:3]
        pc = projected(center,projection,width,height)
        p = projected(v,projection,width,height)
        lengths = np.array([np.linalg.norm(p[(j+1)%3]-p[(j+2)%3]) for j in range(3)])
        inc = lengths@p/lengths.sum()
        e = np.roll(p,-1,axis=0)-p
        normals = np.stack([e[:,1],-e[:,0]],axis=-1)/np.linalg.norm(e,axis=1)[:,None]
        offsets = -np.sum(normals*p,axis=1)
        d = normals@inc+offsets
        sign = np.where(d>0,-1,1)
        normals *= sign[:,None]
        offsets *= sign
        d *= sign
        radius = -d[-1]
        extent = np.linalg.norm(p-pc,axis=1).max()
        normal = view[:3,:3]@np.cross(v[1]-v[0],v[2]-v[0])
        normal /= max(np.linalg.norm(normal),1e-4)
        cos = abs(normal@vc/max(np.linalg.norm(vc),1e-4))
        stop = float(np.float32(.01))
        skip = o < stop or vc[2] <= .2 or cos < .001 or extent>1600 or extent<1 or radius<1
        rmin = np.array([width//16,height//16])
        rmax = np.array([0,0])
        if not skip:
            size = d[0]*(stop/o)**(1/sig)
            adjusted = offsets-size
            for j in range(3):
                ids = [j,(j-1)%3]
                mat = normals[ids]
                if abs(np.linalg.det(mat)) <= .001:
                    continue
                cross = np.linalg.solve(mat,-adjusted[ids])
                # Fixtures keep all tightened intersections nonnegative.
                assert np.all(cross >= 0)
                lo = np.minimum([width//16,height//16],np.floor(cross/16)).astype(int)
                hi = np.minimum([width//16,height//16],np.floor((cross+15)/16)).astype(int)
                rmin, rmax = np.minimum(rmin,lo), np.maximum(rmax,hi)
            skip = np.prod(rmax-rmin) == 0
        dist = pixel@normals.T+offsets
        u = np.maximum(-np.max(dist,axis=-1)/radius,0)
        ideal = o*u**sig
        alpha = np.minimum(.99,ideal)
        alpha[(alpha < 1/255) | (xx//16 < rmin[0]) | (xx//16 >= rmax[0]) |
              (yy//16 < rmin[1]) | (yy//16 >= rmax[1])] = 0
        if skip:
            alpha[:] = 0
        images.append(alpha)
        radii.append(0 if skip else int(np.ceil(extent)))
        records.append({'inradius':float(radius),'center_depth':float(vc[2]),
                        'tile_min':rmin.tolist(),'tile_max':rmax.tolist(),
                        'culled':bool(skip),'ideal_sample_mass':float(ideal.sum())})
    trans = np.ones((height,width))
    done = np.zeros((height,width),bool)
    for j in np.argsort([r['center_depth'] for r in records],kind='stable'):
        candidate = trans*(1-images[j])
        done |= candidate < 1e-4
        trans = np.where(done,trans,candidate)
    return 1-trans, np.array(radii), records
