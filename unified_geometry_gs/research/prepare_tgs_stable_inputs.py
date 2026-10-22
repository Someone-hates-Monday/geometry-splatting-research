"""Freeze v2 inputs using only independent geometry, never native outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tgs_mixed_numpy import evaluate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True);inputs=OUT/'inputs';inputs.mkdir(exist_ok=True)
    path=OUT/'input_plan.json';assert not path.exists()
    old=ROOT/'experiments/tgs_radius_contraction_v1/input_plan.json'
    plan=json.loads(old.read_text())
    new=dict(old_plan_sha256=sha(old),old_cases=plan['old_cases'],stress_cases=plan['stress_cases'],
        holdout_cases=[],protocol_sha256=sha(ROOT/'TGS_STABLE_WINDOW_PROTOCOL_V2.md'),script_sha256=sha(Path(__file__)),training=False)
    # Preserve the exact existing rational input (strip recorded outputs only).
    origin=ROOT/'experiments/tgs_radius_contraction_v1/tile_coverage/native_reproduction.npz'
    with np.load(origin) as z:f={k:z[k].copy() for k in z.files if not k.startswith('native_') and k!='reference_gradient'}
    f.update(kind='rgb',check='smooth')
    target=inputs/'rational.npz';np.savez_compressed(target,**f)
    new['rational_cases']=[dict(name='rational_tile_counterexample',input=target.relative_to(ROOT).as_posix(),sha256=sha(target),origin_sha256=sha(origin))]
    rng=np.random.default_rng(202610202)
    for i in range(12):
        w=[257,2047,8191][i//4];h=193;r=[2.,8.,20.][(i//4+i%4)%3]
        angle=[.37,1.27,2.31,4.58][i%4];co,si=np.cos(angle),np.sin(angle)
        q=(np.array([[0.,0.],[4.,0.],[0.,3.]])-1.)*(r/1.)
        q=q@np.array([[co,-si],[si,co]]).T
        shift=[[.8,h/2],[w-.8,h/2],[w*.37,.8],[w*.61,h-.8]][i%4]
        q+=shift
        z=np.array([1.5,2.3,3.1]);p=np.c_[((2*q+1)/[w,h]-1)*z[:,None],z].astype(np.float32)
        if i%2:p=p[[0,2,1]]
        f=dict(name=f'holdout_{i:02d}',n=1,w=w,h=h,degree=-1,
            theta=np.r_[p.ravel(),[.55,1.7,3.2][i%3],[.1,.65,.98,.012][i%4]].astype(np.float32),
            colors=np.array([[.7,.4,.2]],dtype=np.float32),background=np.array([.08,.13,.21],dtype=np.float32),
            view=np.eye(4,dtype=np.float32),projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]],dtype=np.float32),
            campos=np.zeros(3,dtype=np.float32),tx=1.,ty=1.,kind='rgb',check='smooth')
        xx0=max(0,int(np.floor(q[:,0].min()))-2);xx1=min(w-1,int(np.ceil(q[:,0].max()))+2)
        yy0=max(0,int(np.floor(q[:,1].min()))-2);yy1=min(h-1,int(np.ceil(q[:,1].max()))+2)
        yy,xx=np.mgrid[yy0:yy1+1,xx0:xx1+1];pixels=np.c_[xx.ravel(),yy.ravel()]
        f.update(pixels=pixels,pixel_grad=np.zeros((len(pixels),3),np.float32))
        _,all_rgb,fp=evaluate(f['theta'].astype(float),f)
        alpha=(all_rgb[:,0]-f['background'][0])/(f['colors'][0,0]-f['background'][0])
        eligible=np.array(fp['valid'][0])&(alpha>.0045)&(alpha<.95)
        ids=np.flatnonzero(eligible);assert len(ids)>=4,(i,len(ids))
        rank=lambda j:hashlib.sha256(f'tgs_stable_v2/{i}/{pixels[j,0]}/{pixels[j,1]}'.encode()).hexdigest()
        selected=np.array(sorted(ids,key=rank)[:4])
        rejected=inputs/(f['name']+'_selection.npz')
        np.savez_compressed(rejected,pixels=pixels,eligible=eligible,alpha=alpha,selected=selected,ideal_valid=np.array(fp['valid'][0]))
        patterns=rng.normal(size=(4,4,3));patterns/=np.abs(patterns).max(axis=(1,2),keepdims=True)
        f.update(pixels=pixels[selected],patterns=patterns.astype(np.float32),pixel_grad=patterns[0].astype(np.float32))
        directions=rng.normal(size=(4,2,11));directions/=np.linalg.norm(directions,axis=-1,keepdims=True)
        f['fd_directions']=directions
        _,_,base=evaluate(f['theta'].astype(float),f)
        hh=min(.0005,r/(16*max(w,h)))
        for halves in range(12):
            stable=all(evaluate((f['theta'].astype(float)+s*hh*d).astype(np.float32).astype(float),f)[2]==base for d in directions.reshape(-1,11) for s in [-1,1])
            if stable:break
            hh*=.5
        assert stable
        f['fd_steps']=np.array([hh,.3*hh,.1*hh])
        target=inputs/(f['name']+'.npz');np.savez_compressed(target,**f)
        new['holdout_cases'].append(dict(name=f['name'],input=target.relative_to(ROOT).as_posix(),sha256=sha(target),selection=rejected.relative_to(ROOT).as_posix(),selection_sha256=sha(rejected),rotation=angle,width=w,nominal_inradius=r,depths=z.tolist(),fd_halvings=halves,candidate_pixels=len(pixels),eligible_pixels=len(ids)))
    path.write_text(json.dumps(new,indent=2)+'\n')
    print(json.dumps(dict(old=18,stress=48,rational=1,holdout=12,input_plan_sha256=sha(path),protocol_sha256=new['protocol_sha256'])))


if __name__=='__main__':main()
