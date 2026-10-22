"""Read-only model/camera/case adapters for original and forward-only RGB control."""
import json
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,module,pack,split,colors,CrossWorkload
from run_tgs_real_events import camera_tensors
from run_tgs_local_split_v19 import cache,same_bits
OUT=ROOT/'experiments/tgs_aux_ablation_v21';V19=ROOT/'experiments/tgs_local_split_v19';V20=ROOT/'experiments/tgs_local_stress_v20';V14=ROOT/'experiments/tgs_split_color_v14'

def control():
    r=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(r['path']))==r['sha256'];return module(r['path'])

def view_list(w,scene):
    result=[dict(label=f'old_{i}',cam=c,ci=i) for i,c in enumerate(w.cameras)]
    p=json.loads((ROOT/'experiments/tgs_sh_isolation_v12/input_plan.json').read_text());identity=json.loads((ROOT/'experiments/tgs_sh_isolation_v12/identity_audit.json').read_text())
    row=next(x for x in p['rows'] if x['scene']==scene and x['cohort']=='fresh');ir=next(x for x in identity['rows'] if x['scene']==scene and x['cohort']=='fresh');H,W=ir['shape'][:2]
    result.append(dict(label='fresh',cam=camera_tensors(row['camera'],W,H),ci=None));return result

def cases():
    a=json.loads((V19/'results.json').read_text());b=json.loads((V20/'results.json').read_text());out=[]
    for r in a['rows']:out.append(dict(scene=r['scene'],view=f"old_{r['camera']}",parent_id=r['parent_id'],event=r['event'],appearance='hold',key=f"{r['scene']}_old{r['camera']}_e{r['event']}",prior=r))
    for r in b['rows']:out.append(dict(scene=r['scene'],view='fresh',parent_id=r['parent_id'],parent_index=r['parent_index'],appearance=r['appearance'],key=f"{r['scene']}_fresh_p{r['parent_index']}_{r['appearance']}",prior=r))
    return out

def geometry_equal(a,b):
    assert same_bits(a[6],b[6]);mask=a[6]>0
    for i,(x,y) in enumerate(zip(a,b)):
        if i==6:continue
        assert same_bits(x[mask],y[mask]),i

def rgba_equal(a,b,sa,sb,cam):
    ag,ar,al,ap=cache(a,sa,cam);bg,br,bl,bp=cache(b,sb,cam)
    geometry_equal(ag,bg);assert same_bits(ar,br) and same_bits(al,bl)
    checks=dict(rgb=same_bits(a[1],b[1]),alpha=same_bits(a[2][1],b[2][1]),paths=same_bits(ap,bp))
    assert all(checks.values()),checks
    return checks
