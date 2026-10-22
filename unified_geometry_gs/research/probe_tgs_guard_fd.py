"""Run the unchanged v3 FD corpus against v4's cached forward choices."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from tgs_mixed_numpy import evaluate
from probe_tgs_stable_window import SafeNative,signature
from probe_tgs_compositing_guard import modules
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compositing_guard_v4';OLD=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'fd_results.json';assert not target.exists();assert json.loads((OUT/'results.json').read_text())['all_gates']
    plan=json.loads((OUT/'input_plan_extended.json').read_text());records=[];started=time.time()
    for v,m in modules().items():
        for e in plan['cases']:
            with np.load(ROOT/e['input']) as z:f={k:z[k].copy() for k in z.files}
            if 'fd_steps' not in f:continue
            for k in ['n','w','h','degree']:f[k]=int(f[k])
            name=str(f['name']);native=SafeNative(m,f);base=native.forward(f['theta']);bs=signature(base[1],f)
            with np.load(OLD/('reference_'+name+'.npz')) as z:exact=z['gradients']
            for pi,p in enumerate(f['patterns']):
                f['pixel_grad']=p;_,_,fp=evaluate(f['theta'].astype(float),f)
                for hh in f['fd_steps']:
                    for di,d in enumerate(f['fd_directions'][pi]):
                        xp=(f['theta'].astype(float)+hh*d).astype(np.float32);xm=(f['theta'].astype(float)-hh*d).astype(np.float32)
                        lp,rp,_,_,_,sp=native.forward(xp);lm,rm,_,_,_,sm=native.forward(xm);ps,ms=signature(rp,f),signature(rm,f)
                        cp,_,fpp=evaluate(xp.astype(float),f);cm,_,fpm=evaluate(xm.astype(float),f);actual=(xp.astype(float)-xm.astype(float))/(2*hh)
                        tile=ps['selected_tile_members']==ms['selected_tile_members']==bs['selected_tile_members'];alpha=ps['alpha_support']==ms['alpha_support']==bs['alpha_support'];kernel=fpp==fpm==fp
                        records.append(dict(name=name,variant=v,pattern=pi,step=float(hh),direction=di,actual_direction=actual.tolist(),expected=float(exact[pi]@actual),native_fd=(lp-lm)/(2*hh),cpu_fd=float(cp-cm)/(2*hh),plus_samples=sp.tolist(),minus_samples=sm.tolist(),plus_signature=ps,minus_signature=ms,tile_stable=tile,alpha_stable=alpha,kernel_stable=kernel,smooth=bool(tile and alpha and kernel)))
        print(json.dumps(dict(variant=v,records=len(records))),flush=True)
    target.write_text(json.dumps(dict(records=records,count=len(records),all_support_stable=all(d['smooth'] for d in records),script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'results.json'),seconds=time.time()-started,training=False),indent=2)+'\n')
    print(json.dumps(dict(count=len(records),all_support_stable=all(d['smooth'] for d in records))),flush=True)
if __name__=='__main__':main()
