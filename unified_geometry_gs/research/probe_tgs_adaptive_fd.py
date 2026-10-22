"""Retain the exact 5184 v2 probe steps across adaptive/oracle, no new tuning."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from tgs_mixed_numpy import evaluate
from probe_tgs_stable_window import SafeNative,signature
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'fd_results.json';assert not target.exists()
    results=json.loads((OUT/'results.json').read_text())
    if not results['candidate_gates_passed']:
        target.write_text(json.dumps(dict(executed=False,reason='VJP or complete-image candidate gate failed',source_sha256=sha(OUT/'results.json'),training=False),indent=2)+'\n');print('FD deferred: candidate gate failed');return
    plan=json.loads((OUT/'input_plan.json').read_text());records=[];started=time.time()
    for variant in ['adaptive','oracle64']:
        r=json.loads((OUT/variant/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256']
        spec=importlib.util.spec_from_file_location(p.stem,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        for e in plan['cases']:
            with np.load(ROOT/e['input']) as z:f={k:z[k].copy() for k in z.files}
            if 'fd_steps' not in f:continue
            for k in ['n','w','h','degree']:f[k]=int(f[k])
            name=str(f['name']);native=SafeNative(mod,f);base=native.forward(f['theta']);bs=signature(base[1],f)
            with np.load(OUT/('reference_'+name+'.npz')) as z:exact=z['gradients']
            for pi,pattern in enumerate(f['patterns']):
                f['pixel_grad']=pattern;_,_,fp=evaluate(f['theta'].astype(float),f)
                for hh in f['fd_steps']:
                    for di,d in enumerate(f['fd_directions'][pi]):
                        xp=(f['theta'].astype(float)+hh*d).astype(np.float32);xm=(f['theta'].astype(float)-hh*d).astype(np.float32)
                        lp,rp,_,_,_,sp=native.forward(xp);lm,rm,_,_,_,sm=native.forward(xm)
                        cp,_,fpp=evaluate(xp.astype(float),f);cm,_,fpm=evaluate(xm.astype(float),f)
                        ps,ms=signature(rp,f),signature(rm,f);actual=(xp.astype(float)-xm.astype(float))/(2*hh)
                        tile=ps['selected_tile_members']==ms['selected_tile_members']==bs['selected_tile_members'];alpha=ps['alpha_support']==ms['alpha_support']==bs['alpha_support']
                        kernel=fpp==fpm==fp and bool((rp[3]>0).all()) and bool((rm[3]>0).all())
                        records.append(dict(name=name,variant=variant,pattern=pi,step=float(hh),direction=di,actual_direction=actual.tolist(),expected=float(exact[pi]@actual),native_fd=(lp-lm)/(2*hh),cpu_fd=float(cp-cm)/(2*hh),plus_samples=sp.tolist(),minus_samples=sm.tolist(),plus_signature=ps,minus_signature=ms,tile_stable=tile,alpha_stable=alpha,kernel_stable=bool(kernel),smooth=bool(tile and alpha and kernel)))
            print(json.dumps(dict(variant=variant,name=name,records_so_far=len(records))),flush=True)
    final=dict(executed=True,records=records,count=len(records),all_support_stable=all(r['smooth'] for r in records),script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'results.json'),seconds=time.time()-started,training=False)
    target.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({k:v for k,v in final.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
