"""Read-only frozen inputs; CPU arithmetic ablations, not bitwise CUDA emulation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tgs_mixed_numpy import evaluate

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'experiments/tgs_radius_contraction_v1'
OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def project(f,dtype):
    p=f['theta'][:9].reshape(3,3).astype(dtype)
    hom=np.c_[p,np.ones(3,dtype=dtype)]@f['projection'].astype(dtype).T
    recip=dtype(1)/(hom[:,3:]+dtype(1e-7))
    return ((hom[:,:2]*recip+dtype(1))*np.array([f['w']/2,f['h']/2],dtype=dtype)-dtype(.5)).astype(dtype)


def window(q,f,dtype,local=False,area_radius=False):
    anchor=np.floor(q.astype(float).mean(0)) if local else np.zeros(2)
    v=(q.astype(float)-anchor).astype(dtype)
    pix=(f['pixels']-anchor).astype(dtype)
    e=np.roll(v,-1,0)-v;length=np.sqrt((e*e).sum(1))
    a,b,c=length[1],length[2],length[0]
    inc=(a*v[0]+b*v[1]+c*v[2])/(a+b+c)
    normals=np.c_[e[:,1],-e[:,0]]/length[:,None]
    off=-(normals*v).sum(1)
    dist=(normals*inc).sum(1)+off
    flip=np.where(dist>0,dtype(-1),dtype(1))
    normals*=flip[:,None];off*=flip
    D=e[0,0]*(v[2,1]-v[0,1])-e[0,1]*(v[2,0]-v[0,0])
    r=abs(D)/length.sum() if area_radius else -dist[-1]*flip[-1]
    phi=(pix@normals.T+off).max(1)
    sig,op=f['theta'][9:11].astype(dtype)
    alpha=np.minimum(dtype(.99),op*np.maximum(dtype(0),-phi/r)**sig)
    alpha=np.where((phi<=0)&(alpha>=dtype(1/255)),alpha,dtype(0))
    return alpha[:,None]*f['colors'][0].astype(dtype)+(dtype(1)-alpha[:,None])*f['background'].astype(dtype)


def main():
    OUT.mkdir(exist_ok=True)
    target=OUT/'cpu_precision_attribution.json';assert not target.exists()
    plan=json.loads((OLD/'input_plan.json').read_text());rows=[]
    for entry in plan['stress_cases']:
        path=ROOT/entry['input'];assert sha(path)==entry['sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for k in ['w','h','n','degree']:f[k]=int(f[k])
        _,ref,_=evaluate(f['theta'].astype(float),f)
        with np.load(OLD/('piecewise_'+entry['name']+'.npz')) as z:native=z['native_samples']
        q64,q32=project(f,np.float64),project(f,np.float32)
        stages=dict(q32_geometry64=window(q32,f,np.float64),
            q64_global32=window(q64,f,np.float32),q32_global32=window(q32,f,np.float32),
            q64_local32_lastdist=window(q64,f,np.float32,True),
            q64_local32_area=window(q64,f,np.float32,True,True))
        rows.append(dict(name=entry['name'],width=f['w'],native_error=float(np.abs(native-ref).max()),
            projected_q32_error=float(np.abs(q32-q64).max()),
            stages={k:dict(reference_error=float(np.abs(v-ref).max()),native_difference=float(np.abs(v-native).max())) for k,v in stages.items()}))
    result=dict(input_plan_sha256=sha(OLD/'input_plan.json'),script_sha256=sha(Path(__file__)),
        caveat='NumPy float32 is an arithmetic ablation without CUDA FMA/fast-math/binning; not an exact causal decomposition of native errors.',
        rows=rows,summary={k:max(r['stages'][k]['reference_error'] for r in rows) for k in stages},training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(summary=result['summary'],failed_native=[r for r in rows if r['native_error']>3e-5])),flush=True)


if __name__=='__main__':main()
