"""Freeze ALL previously counted empirical-bound violations before any detailed replay."""
import json
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
from audit_tgs_window_scene_v15 import reconstruct
OUT=ROOT/'experiments/tgs_window_scene_v15';OLD=ROOT/'experiments/tgs_split_color_v14'

def main():
    audit=json.loads((OUT/'independent_audit.json').read_text());result=json.loads((OUT/'results.json').read_text())
    target=OUT/'violation_pixels.json';assert not target.exists();pixels=[]
    for row in audit['rows']:
        if row['fixed_slot_bound_violating_pixels']==0:continue
        case=next(r for r in result['cases'] if all(r[k]==row[k] for k in ['scene','event','camera']))
        base=OUT/row['scene']/f"event_{row['event']}"/f"camera_{row['camera']}"
        P=reconstruct(base/'parent.npz',case['versions']['parent'])['alpha'].astype(float)
        Q=reconstruct(base/'children.npz',case['versions']['children'])['alpha'].astype(float)
        old=OLD/row['scene']/f"event_{row['event']}"/f"camera_{row['camera']}"
        raw=np.load(old/'changes.npz');x0,y0,x1,y1=raw['bbox'];delta=Q-P
        G=raw['hold_rgb'].astype(float)-raw['before_rgb'].astype(float)
        Cmax=np.array(row['fixed_slot_Cmax']);bound=Cmax[:,None,None]*np.abs(delta[y0:y1,x0:x1])[None]
        yy,xx=np.nonzero(np.any(np.abs(G)>bound+3e-5,axis=0))
        assert len(xx)==row['fixed_slot_bound_violating_pixels']
        for y,x in zip(yy,xx):
            pixels.append(dict(scene=row['scene'],event=row['event'],camera=row['camera'],parent_id=row['parent_id'],
                pixel=[int(x+x0),int(y+y0)],parent_alpha=float(P[y+y0,x+x0]),child_alpha=float(Q[y+y0,x+x0]),
                G=G[:,y,x].tolist(),conditional_bound=bound[:,y,x].tolist(),max_excess=float((np.abs(G[:,y,x])-bound[:,y,x]).max())))
    assert len(pixels)==3
    write_new(target,dict(pixels=pixels,selection='All three already-counted fixed-slot empirical bound violation pixels; no subsampling',
        source_audit_sha256=sha(OUT/'independent_audit.json'),source_render_sha256=sha(OUT/'results.json'),
        script_sha256=sha(Path(__file__)),posthoc_mechanism_trace=True,training=False))
    print(json.dumps(pixels),flush=True)

if __name__=='__main__':main()
