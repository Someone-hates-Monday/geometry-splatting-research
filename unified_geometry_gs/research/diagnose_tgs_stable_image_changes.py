"""Post-gate exploratory full-image CPU comparison on 18 retained old cases.

No new pass gate: pixelwise cutoff/early-stop/tie decisions can be discontinuous.
Report every pixel and the maximum discrepancy without deleting boundary pixels.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from tgs_mixed_numpy import evaluate

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'full_old_image_diagnosis.json';assert not target.exists()
    plan=json.loads((OUT/'input_plan.json').read_text());rows=[]
    for e in plan['old_cases']:
        path=ROOT/e['input'];assert sha(path)==e['sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        with np.load(OUT/('direct_'+e['name']+'.npz')) as z:new=z['native_rgb']
        with np.load(ROOT/e['old_output']) as z:old=z['native_rgb']
        yy,xx=np.mgrid[:f['h'],:f['w']];pix=np.c_[xx.ravel(),yy.ravel()];chunks=[]
        for k in range(0,len(pix),16384):
            selected=pix[k:k+16384];ff=dict(f,pixels=selected,pixel_grad=np.zeros((len(selected),3)))
            chunks.append(evaluate(f['theta'].astype(float),ff)[1])
        ideal=np.concatenate(chunks).reshape(f['h'],f['w'],3).transpose(2,0,1)
        ne=np.abs(new-ideal);oe=np.abs(old-ideal);delta=np.abs(new-old)
        c,y,x=np.unravel_index(ne.argmax(),ne.shape)
        dc,dy,dx=np.unravel_index(delta.argmax(),delta.shape)
        where=np.unique(np.array([[x,y],[dx,dy]]),axis=0)
        ff=dict(f,pixels=where,pixel_grad=np.zeros((len(where),3)))
        _,_,fp=evaluate(f['theta'].astype(float),ff)
        row=dict(name=e['name'],n=f['n'],width=f['w'],height=f['h'],max_new_error=float(ne.max()),max_old_error=float(oe.max()),mean_new_error=float(ne.mean()),mean_old_error=float(oe.mean()),pixels_new_error_gt_3e5=int((ne.max(0)>3e-5).sum()),pixels_changed_gt_3e5=int((delta.max(0)>3e-5).sum()),max_historical_delta=float(delta.max()),max_new_error_pixel=[int(x),int(y)],max_delta_pixel=[int(dx),int(dy)],inspected_pixels=where.tolist(),ideal_fingerprint=fp,new_rgb=[new[:,py,px].tolist() for px,py in where],old_rgb=[old[:,py,px].tolist() for px,py in where],ideal_rgb=[ideal[:,py,px].tolist() for px,py in where])
        rows.append(row)
        np.savez_compressed(OUT/('full_ideal_'+e['name']+'.npz'),ideal_rgb=ideal)
        print(json.dumps({k:row[k] for k in ['name','max_new_error','max_old_error','pixels_new_error_gt_3e5','max_historical_delta']}),flush=True)
    result=dict(input_plan_sha256=sha(OUT/'input_plan.json'),script_sha256=sha(Path(__file__)),exploratory=True,selection='All 18 historical cases and every image pixel; no exclusions. Worst pixels selected after output for diagnosis.',rows=rows,training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
