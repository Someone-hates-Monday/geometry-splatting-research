"""Freeze all prior cases, 48 geometric stresses and eight signed RGB VJPs.

Selection uses only geometry and branch safety, before native candidate output.
Every rejected lattice sample is retained as a mask in compressed NPZ.
"""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from tgs_mixed_numpy import evaluate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_radius_contraction_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def old_cases():
    base=ROOT/'experiments/tgs_backward_v1/geometry_fixed'
    for row in json.loads((base/'results.json').read_text())['fixtures']:
        with np.load(base/(row['name']+'.npz')) as z:a={k:z[k].copy() for k in z.files}
        f=dict(name='single_'+row['name'],n=1,w=int(a['w']),h=int(a['h']),degree=-1,theta=a['theta'],
               pixels=a['pixels'],pixel_grad=(a['weights'][:,None]*a['channel_weights']).astype(np.float32),
               view=a['view'],projection=a['projection'],colors=a['color'][None],background=np.zeros(3),campos=np.zeros(3),
               tx=1.,ty=1.,kind='rgb',check='smooth')
        yield f,ROOT/('experiments/tgs_branch_aux_v1/regression_'+f['name']+'.npz')
    for relative in ['experiments/tgs_mixed_backward_v1','experiments/tgs_real_backward_v2']:
        base=ROOT/relative
        for row in json.loads((base/'results.json').read_text())['records']:
            if row['variant']!='fixed':continue
            with np.load(base/(row['name']+'_input.npz')) as z:f={k:z[k].copy() for k in z.files if k not in ['probe_directions','probe_names']}
            for k in ['n','w','h','degree']:f[k]=int(f[k])
            f.update(name=row['name'],kind='rgb',check='smooth')
            yield f,ROOT/('experiments/tgs_branch_aux_v1/regression_'+f['name']+'.npz')
    base=ROOT/'experiments/tgs_branch_aux_v1'
    for row in json.loads((base/'piecewise_results.json').read_text())['rows']:
        with np.load(base/(row['name']+'_input.npz')) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        yield f,base/('piecewise_'+f['name']+'.npz')


def stress_cases():
    rng=np.random.default_rng(142909)
    grid=list(itertools.product([256,2048,8192],[1.25,2.,8.,40.],[1.,4.,8.],[.1,.5,.9],[False,True]))
    chosen=rng.choice(len(grid),size=48,replace=False)
    h=256
    for i,index in enumerate(chosen):
        w,r,aspect,position,reverse=grid[index]
        short=2*r/(aspect+1-np.sqrt(aspect*aspect+1));long=aspect*short
        inc=np.array([w*position+rng.uniform(-.4,.4),h/2+rng.uniform(-.4,.4)])
        q=np.array([inc-[r,r],inc-[r,r]+[long,0],inc-[r,r]+[0,short]])
        z=np.full(3,2.)
        points=np.c_[((2*q+1)/[w,h]-1)*z[:,None],z].astype(np.float32)
        if reverse:points=points[[0,2,1]]
        f=dict(name=f'stress_{i:02d}',n=1,w=w,h=h,degree=-1,theta=np.r_[points.ravel(),1.3,.65].astype(np.float32),
               view=np.eye(4,dtype=np.float32),projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]],dtype=np.float32),
               colors=np.array([[.7,.4,.2]],dtype=np.float32),background=np.array([.08,.13,.21],dtype=np.float32),
               campos=np.zeros(3,dtype=np.float32),tx=1.,ty=1.,kind='rgb',check='smooth')
        actual=(points[:,:2].astype(float)/(points[:,2:3].astype(float)+1e-7)+1)*[w/2,h/2]-.5
        e=np.roll(actual,-1,0)-actual;length=np.linalg.norm(e,axis=1)
        D=np.linalg.det(np.array([actual[1]-actual[0],actual[2]-actual[0]]));rr=abs(D)/length.sum()
        n=np.c_[e[:,1],-e[:,0]]/length[:,None]*np.sign(D);b=-(n*actual).sum(1)
        x0=max(0,int(np.ceil(actual[:,0].min())));x1=min(w-1,int(np.floor(actual[:,0].max())))
        y0=max(0,int(np.ceil(actual[:,1].min())));y1=min(h-1,int(np.floor(actual[:,1].max())))
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1];pixels=np.c_[xx.ravel(),yy.ravel()]
        phi=pixels@n.T+b;ordered=np.sort(phi,axis=1)
        u=-ordered[:,-1]/rr
        eligible=(ordered[:,-1]<-.1*rr)&((ordered[:,-1]-ordered[:,-2])>.05*rr)&(u>0)&(u<1)
        eligible&=(.65*np.maximum(u,0)**1.3>.02)
        eligible_ids=np.flatnonzero(eligible)
        assert len(eligible_ids)>=4,(i,len(eligible_ids))
        rank=lambda j:hashlib.sha256(f'tgs_contraction_v1/{i}/{pixels[j,0]}/{pixels[j,1]}'.encode()).hexdigest()
        selected=sorted(eligible_ids,key=rank)[:4]
        f['pixels']=pixels[selected]
        patterns=[];fd_directions=[]
        for vi in range(8):
            vrng=np.random.default_rng(142909+vi)
            weights=vrng.normal(size=(4,3));weights[0,0]=abs(weights[0,0])+.2;weights[1,0]=-abs(weights[1,0])-.2
            weights/=np.abs(weights).max();patterns.append(weights.astype(np.float32))
            dense=vrng.normal(size=11);dense/=np.linalg.norm(dense)
            j=vi if vi<6 else (9 if vi==6 else 10)
            fd_directions.append([np.eye(1,11,j)[0],dense])
        f['patterns']=np.array(patterns);f['pixel_grad']=f['patterns'][0]
        f['fd_directions']=np.array(fd_directions)
        # Geometry-only branch preservation; weights never enter visibility.
        _,_,fp=evaluate(f['theta'].astype(float),f)
        initial_h=min(.003,rr/(8*max(w,h)))
        for halves in range(9):
            hh=initial_h*2**(-halves);stable=True
            for dirs in f['fd_directions']:
                for d in dirs:
                    for sign in [-1,1]:
                        x=(f['theta'].astype(float)+sign*hh*d).astype(np.float32)
                        _,_,other=evaluate(x.astype(float),f)
                        if other!=fp:stable=False;break
                    if not stable:break
                if not stable:break
            if stable:break
        assert stable,'No branch-safe representable perturbation in the frozen budget'
        f['fd_steps']=np.array([hh,.3*hh,.1*hh])
        metadata=dict(grid_index=int(index),width=w,target_inradius=r,actual_inradius=rr,aspect=aspect,
                      position=position,reverse=reverse,anchor=inc.tolist(),depths=z.tolist(),
                      selected_candidate_indices=[int(j) for j in selected],candidate_pixels=len(pixels),
                      eligible_pixels=len(eligible_ids),fd_initial_step=initial_h,fd_halvings=halves,
                      branch_fingerprint=fp)
        rejection=dict(pixels=pixels,eligible_mask=eligible,phi=phi,selected_indices=np.array(selected))
        yield f,metadata,rejection


def main():
    target=OUT/'input_plan.json';assert not target.exists()
    inputs=OUT/'inputs';inputs.mkdir(exist_ok=True)
    plan=dict(protocol_sha256=sha(ROOT/'TGS_RADIUS_CONTRACTION_PROTOCOL_V1.md'),
              script_sha256=sha(Path(__file__)),old_cases=[],stress_cases=[],stress_vjp_seeds=list(range(142909,142917)),
              clarification='H=256; right-triangle leg ratio defines aspect; all base depths=2; winding reversed by swapping vertices; no candidate outcomes used',
              geometry_seed=142909,training=False)
    for f,image in old_cases():
        path=inputs/('old_'+f['name']+'.npz');np.savez_compressed(path,**f)
        plan['old_cases'].append(dict(name=f['name'],input=path.relative_to(ROOT).as_posix(),sha256=sha(path),
                                     old_output=image.relative_to(ROOT).as_posix(),old_output_sha256=sha(image)))
    assert len(plan['old_cases'])==18
    for f,metadata,rejection in stress_cases():
        path=inputs/(f['name']+'.npz');np.savez_compressed(path,**f)
        rp=inputs/(f['name']+'_selection.npz');np.savez_compressed(rp,**rejection)
        plan['stress_cases'].append(dict(name=f['name'],input=path.relative_to(ROOT).as_posix(),sha256=sha(path),
            selection=rp.relative_to(ROOT).as_posix(),selection_sha256=sha(rp),metadata=metadata))
    assert len(plan['stress_cases'])==48
    target.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(old_cases=18,stress_cases=48,signed_vjps_per_stress=8,plan_sha256=sha(target))),flush=True)


if __name__=='__main__':main()
