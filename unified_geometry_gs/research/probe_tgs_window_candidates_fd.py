"""The unchanged5184 retained probe steps, on lean guard and oracle only."""
import json,time
from pathlib import Path
import numpy as np
from tgs_window_candidates_common import ROOT,OUT,BASES,sha,modules
from tgs_mixed_numpy import evaluate
from probe_tgs_stable_window import SafeNative,signature
def main():
    target=OUT/'fd_results.json';assert not target.exists();assert json.loads((OUT/'case_results.json').read_text())['candidate_gates']['oracle']
    plan=json.loads((OUT/'input_plan.json').read_text());mods=modules();records=[];started=time.time()
    for v in BASES:
        for e in plan['cases']:
            with np.load(ROOT/e['input']) as z:f={k:z[k].copy() for k in z.files}
            if 'fd_steps' not in f:continue
            for k in ['n','w','h','degree']:f[k]=int(f[k])
            name=str(f['name']);native=SafeNative(mods[v],f);base=native.forward(f['theta']);bs=signature(base[1],f)
            with np.load(ROOT/('experiments/tgs_adaptive_window_v3/reference_'+name+'.npz')) as z:exact=z['gradients']
            for pi,p in enumerate(f['patterns']):
                f['pixel_grad']=p;_,_,fp=evaluate(f['theta'].astype(float),f)
                for hh in f['fd_steps']:
                    for di,d in enumerate(f['fd_directions'][pi]):
                        xp=(f['theta'].astype(float)+hh*d).astype(np.float32);xm=(f['theta'].astype(float)-hh*d).astype(np.float32);lp,rp,_,_,_,sp=native.forward(xp);lm,rm,_,_,_,sm=native.forward(xm);ps,ms=signature(rp,f),signature(rm,f)
                        cp,_,fpp=evaluate(xp.astype(float),f);cm,_,fpm=evaluate(xm.astype(float),f);actual=(xp.astype(float)-xm.astype(float))/(2*hh)
                        tile=ps['selected_tile_members']==ms['selected_tile_members']==bs['selected_tile_members'];alpha=ps['alpha_support']==ms['alpha_support']==bs['alpha_support'];kernel=fpp==fpm==fp
                        records.append(dict(name=name,variant=v,pattern=pi,step=float(hh),direction=di,actual_direction=actual.tolist(),expected=float(exact[pi]@actual),native_fd=(lp-lm)/(2*hh),cpu_fd=float(cp-cm)/(2*hh),plus_samples=sp.tolist(),minus_samples=sm.tolist(),plus_signature=ps,minus_signature=ms,tile_stable=tile,alpha_stable=alpha,kernel_stable=kernel,smooth=bool(tile and alpha and kernel)))
        print(json.dumps(dict(variant=v,records=len(records))),flush=True)
    result=dict(records=records,count=len(records),all_support_stable=all(r['smooth'] for r in records),script_sha256=sha(Path(__file__)),case_results_sha256=sha(OUT/'case_results.json'),seconds=time.time()-started,training=False)
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
