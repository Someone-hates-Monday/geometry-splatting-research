"""Freeze old95 and CPU-constructed threshold-adjacent compositing cases."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compositing_guard_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);(OUT/'inputs').mkdir(exist_ok=True);target=OUT/'input_plan.json';assert not target.exists()
    old=ROOT/'experiments/tgs_adaptive_window_v3/input_plan.json';plan=json.loads(old.read_text());cases=plan['cases'].copy();rng=np.random.default_rng(202610205)
    for family in ['tail','median']:
        for shift in [-1,0,1,4]:
            for reverse in [False,True]:
                name=f'compose_{family}_{shift}_{int(reverse)}';desired=np.array([.9,.9,.9,.9,.3] if family=='tail' else [.5,.4,.3]);n=len(desired)
                screen=np.array([[32.,32.],[192.,32.],[32.,152.]])
                z=2.**np.arange(1,n+1);points=np.array([np.c_[((2*screen+1)/256-1)*zz,np.full(3,zz)] for zz in z],np.float32)
                if reverse:points=points[:,[0,2,1]]
                q=(points[...,:2].astype(float)/(points[...,2:]+1e-7)+1)*128-.5
                e=np.roll(q,-1,1)-q;length=np.linalg.norm(e,axis=-1);D=np.linalg.det(np.stack([q[:,1]-q[:,0],q[:,2]-q[:,0]],1))
                cross=e[...,0]*(72-q[...,1])-e[...,1]*(76-q[...,0]);u=(cross*np.sign(D)[:,None]*length.sum(1)[:,None]/(abs(D)[:,None]*length)).min(1)
                sigma=np.full(n,.13,np.float32);op=(desired/u**sigma).astype(np.float32);assert (op<1).all()
                j=3 if family=='tail' else 0
                for _ in range(abs(shift)):op[j]=np.nextafter(op[j],np.float32(np.inf if shift>0 else -np.inf))
                patterns=rng.normal(size=(4,4,3));patterns/=np.abs(patterns).max(axis=(1,2),keepdims=True)
                f=dict(name=name,n=n,w=256,h=256,degree=-1,theta=np.r_[points.ravel(),sigma,op].astype(np.float32),pixels=np.array([[76,72],[79,70],[90,65],[110,60]]),patterns=patterns.astype(np.float32),pixel_grad=patterns[0].astype(np.float32),
                    colors=rng.uniform(.1,.9,(n,3)).astype(np.float32),background=np.zeros(3,np.float32),campos=np.zeros(3,np.float32),view=np.eye(4,dtype=np.float32),projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]],np.float32),tx=1.,ty=1.,kind='rgb',check='smooth')
                path=OUT/'inputs'/(name+'.npz');np.savez_compressed(path,**f);cases.append(dict(name=name,input=path.relative_to(ROOT).as_posix(),sha256=sha(path),family='compositing_cases',opacity_ulp_shift=shift,desired_alpha=desired.tolist()))
    target.write_text(json.dumps(dict(cases=cases,prior_plan_sha256=sha(old),protocol_sha256=sha(ROOT/'TGS_COMPOSITING_GUARD_PROTOCOL_V4.md'),script_sha256=sha(Path(__file__)),training=False),indent=2)+'\n');print(json.dumps(dict(cases=len(cases),plan_sha256=sha(target))))
if __name__=='__main__':main()
