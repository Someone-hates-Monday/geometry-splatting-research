"""Classify the failed recapture by defined output vs invisible scratch, no threshold waiver."""
import gc,json
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,OUT,sha,modules,CrossWorkload,decode,state
from audit_tgs_compensated_baseline_diagnostic import anchor_view,rects

def main():
    target=OUT/'recapture_diagnostic.json';assert not target.exists();plan=json.loads((OUT/'real_plan.json').read_text());inputs=json.loads((OUT/'input_audit.json').read_text());mods=modules();rows=[]
    for scene in [s['scene'] for s in inputs['scenes']]:
        w=CrossWorkload(scene);assert w.input_hashes()==plan['model_inputs'][scene]
        for ci,cam in enumerate(w.camera_metadata):
            r=w.forward(mods['oracle'],ci);torch.cuda.synchronize();_,errors,C=decode(r,w.n,'oracle');an=anchor_view(r,w.n);mn,mx=rects(r,w.n);rad=r[3].cpu().numpy();flags,ids,pref=state(r,dict(n=w.n,w=w.cameras[ci]['w'],h=w.cameras[ci]['h']),'oracle');fields={}
            for key,arr in [('rgb',r[1].cpu().numpy()),('aux',r[2].cpu().numpy()),('radii',rad),('maxblend',r[9].cpu().numpy()),('contributors',ids),('double_coeff',C),('cache_errors',errors),('anchors',an),('rect_min',mn),('rect_max',mx)]:
                old=np.load(OUT/scene/f'camera_{ci}'/'oracle'/(key+'.npy'),mmap_mode='r');assert arr.shape==old.shape and arr.dtype==old.dtype;x=np.ascontiguousarray(arr).view(np.uint8).reshape(arr.shape[0],-1);y=np.ascontiguousarray(old).view(np.uint8).reshape(arr.shape[0],-1);differ=(x!=y).any(1);q=dict(numeric_equal=bool(np.array_equal(arr,old)),literal_bits_equal=not bool(differ.any()),first_axis_mismatch_count=int(differ.sum()))
                if key in ['radii','maxblend','double_coeff','cache_errors','anchors','rect_min','rect_max']:
                    visible=rad>0;bad=np.flatnonzero(differ);q.update(visible_mismatches=int((differ&visible).sum()),invisible_mismatches=int((differ&~visible).sum()),visible_count=int(visible.sum()),first_mismatch_ids=bad[:16].tolist());q['sample_bytes']=[dict(id=int(i),new=x[i].tobytes().hex(),old=y[i].tobytes().hex(),visible=bool(visible[i])) for i in bad[:4]]
                fields[key]=q
            row=dict(scene=scene,camera=ci,name=cam['img_name'],fields=fields);rows.append(row);print(json.dumps(dict(scene=scene,camera=ci,mismatches={k:{q:v for q,v in f.items() if q not in ['sample_bytes','first_mismatch_ids']} for k,f in fields.items() if not f['literal_bits_equal']})),flush=True);del r,C,errors,an,mn,mx;gc.collect()
        assert w.input_hashes()==plan['model_inputs'][scene];del w;gc.collect();torch.cuda.empty_cache()
    only_scratch=all(f['literal_bits_equal'] or (key in ['anchors','rect_min','rect_max'] and f['visible_mismatches']==0) for row in rows for key,f in row['fields'].items());target.write_text(json.dumps(dict(rows=rows,only_invisible_projection_scratch_differs=only_scratch,original_audit_failure_sha256=sha(OUT/'audit_failure.txt'),script_sha256=sha(Path(__file__)),training=False,timing=False),indent=2)+'\n');print(json.dumps(dict(only_invisible_projection_scratch_differs=only_scratch)),flush=True)

if __name__=='__main__':main()
