"""Independent auxiliary-map reference using pairwise distortion and complex step."""
import numpy as np
from tgs_mixed_numpy import evaluate as rgb_evaluate


def evaluate(theta,f):
    n=int(f['n']);S=len(f['pixels']);p=theta[:9*n].reshape(n,3,3)
    sigma,opacity=theta[9*n:10*n],theta[10*n:11*n]
    alphas=[];single_fps=[]
    for i in range(n):
        single=dict(f,n=1,degree=-1,colors=np.ones((1,3)),background=np.zeros(3),pixel_grad=np.zeros((S,3)))
        single.pop('preprocess_visible',None)
        t=np.r_[p[i].ravel(),sigma[i],opacity[i]]
        _,rgb,fp=rgb_evaluate(t,single)
        alphas.append(rgb[:,0]);single_fps.append(fp)
    alphas=np.array(alphas)
    center=np.c_[p.mean(1),np.ones(n)]@f['view'].astype(float).T
    depth=center[:,2]
    cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])@f['view'][:3,:3].astype(float).T
    norm=np.sqrt((cross*cross).sum(-1))
    floor=norm.real<=1e-4
    normals=cross/np.where(floor,1e-4,norm)[:,None]
    flip=(normals*center[:,:3]).sum(-1).real>0
    normals=normals*np.where(flip,-1,1)[:,None]
    order=depth.real.argsort().tolist()
    T=np.ones(S,dtype=theta.dtype);done=np.zeros(S,dtype=bool)
    D=np.zeros(S,dtype=theta.dtype);N=np.zeros((S,3),dtype=theta.dtype);median=np.zeros_like(D)
    all_weights=[]
    fp=dict(single=single_fps,order=order,normal_floor=floor.tolist(),normal_flip=flip.tolist(),
            included=[],terminated=[],median_picks=[])
    for i in order:
        valid=alphas[i].real>0
        stop=valid&(np.real(T*(1-alphas[i]))<1e-4)&~done
        take=valid&~done&~stop
        a=np.where(take,alphas[i],0);weight=T*a
        choose=take&(T.real>.5)
        D+=depth[i]*weight;N+=weight[:,None]*normals[i]
        median=np.where(choose,depth[i],median)
        all_weights.append(weight)
        T*=1-a;done|=stop
        fp['included'].append(take.tolist());fp['terminated'].append(stop.tolist());fp['median_picks'].append(choose.tolist())
    # Pairwise formula avoids reusing the native/Torch moment recurrence.
    distortion=np.zeros(S,dtype=theta.dtype)
    m=100/99.8*(1-.2/depth)
    for i in range(n):
        for j in range(i):
            distortion+=all_weights[i]*all_weights[j]*(m[order[i]]-m[order[j]])**2
    aux=np.stack([D,1-T,N[:,0],N[:,1],N[:,2],median,distortion],-1)
    return (aux*f['aux_grad'].astype(float)).sum(),aux,fp


def complex_gradient(theta,f):
    result=[]
    for j in range(len(theta)):
        x=theta.astype(complex);x[j]+=1e-20j
        result.append(evaluate(x,f)[0].imag/1e-20)
    return np.array(result)
