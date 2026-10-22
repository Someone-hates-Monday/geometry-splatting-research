"""Freeze all v2 inputs plus deterministic boundary exponent tests."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);(OUT/'inputs').mkdir(exist_ok=True)
    target=OUT/'input_plan.json';assert not target.exists()
    old=ROOT/'experiments/tgs_stable_window_v2/input_plan.json';p=json.loads(old.read_text())
    cases=[dict(e,family=family) for family in ['old_cases','stress_cases','rational_cases','holdout_cases'] for e in p[family]]
    rng=np.random.default_rng(202610204)
    for wi,w in enumerate([256,8192]):
        for si,sigma in enumerate([.01,.02,.1,.5]):
            for reverse in [False,True]:
                name=f'boundary_{wi}_{si}_{int(reverse)}';shift=w-256
                q=np.array([[32+shift,32],[192+shift,32],[32+shift,152]],dtype=float)
                points=np.c_[((2*q+1)/[w,256]-1)*2,np.full(3,2.)].astype(np.float32)
                if reverse:points=points[[0,2,1]]
                patterns=rng.normal(size=(4,4,3));patterns/=np.abs(patterns).max(axis=(1,2),keepdims=True)
                f=dict(name=name,n=1,w=w,h=256,degree=-1,theta=np.r_[points.ravel(),sigma,.9999].astype(np.float32),
                    pixels=np.array([[176+shift,44],[188+shift,35],[76+shift,72],[79+shift,70]]),patterns=patterns.astype(np.float32),pixel_grad=patterns[0].astype(np.float32),
                    colors=np.array([[.7,.4,.2]],np.float32),background=np.array([.08,.13,.21],np.float32),campos=np.zeros(3,np.float32),
                    projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0]],np.float32),view=np.eye(4,dtype=np.float32),tx=1.,ty=1.,kind='rgb',check='smooth')
                path=OUT/'inputs'/(name+'.npz');np.savez_compressed(path,**f)
                cases.append(dict(name=name,input=path.relative_to(ROOT).as_posix(),sha256=sha(path),family='boundary_cases'))
    target.write_text(json.dumps(dict(cases=cases,prior_plan_sha256=sha(old),protocol_sha256=sha(ROOT/'TGS_ADAPTIVE_WINDOW_PROTOCOL_V3.md'),script_sha256=sha(Path(__file__)),training=False),indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),plan_sha256=sha(target))))
if __name__=='__main__':main()
