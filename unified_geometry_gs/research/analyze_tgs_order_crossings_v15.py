"""Exact centroid/swap identities and all active crossings on the frozen3 trace pixels."""
import json
from pathlib import Path
import numpy as np
import sympy as s
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_window_scene_v15/trace'

def main():
    target=OUT/'active_crossings.json';assert not target.exists()
    z=s.symbols('za zb zc',real=True);g=sum(z)/3
    children=[(z[0]+(z[0]+z[1])/2+(z[0]+z[2])/2)/3,
              (z[1]+(z[0]+z[1])/2+(z[1]+z[2])/2)/3,
              (z[2]+(z[0]+z[2])/2+(z[1]+z[2])/2)/3,
              ((z[0]+z[1])/2+(z[0]+z[2])/2+(z[1]+z[2])/2)/3]
    assert all(s.expand(children[i]-g-(z[i]-g)/2)==0 for i in range(3)) and s.expand(children[3]-g)==0
    a,b,c,d=s.symbols('a b c d',real=True)
    assert s.expand((a*c+(1-a)*b*d)-(b*d+(1-b)*a*c)-a*b*(c-d))==0
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_verified'];rows=[]
    for r in audit['rows']:
        event=r['event'];x,y=r['pixel'];data={}
        for mode in ['after_actual','after_group']:
            zfile=np.load(OUT/f'e{event}_{x}_{y}_{mode}.npz');tr=zfile['trace'];tr=tr[np.isfinite(tr[:,0])]
            packet=zfile['packed'];rank={int(p[16]):i for i,p in enumerate(packet)}
            data[mode]=(packet,tr,rank)
        packet,tr,actual=data['after_actual'];group=data['after_group'][2]
        active={int(t[0]):i for i,t in enumerate(tr) if t[8]}
        children=[pid for pid in active if pid>=2099999]
        crossings=[]
        for child in children:
            ci=active[child]
            for other,oi in active.items():
                if other>=2099999:continue
                if (actual[child]<actual[other])!=(group[child]<group[other]):
                    crossings.append(dict(child_native_id=child,external_native_id=other,
                        actual_child_rank=actual[child],actual_external_rank=actual[other],
                        fixed_group_child_rank=group[child],fixed_group_external_rank=group[other],
                        child_alpha=float(tr[ci,4]),external_alpha=float(tr[oi,4]),
                        child_cached_center_depth=float(packet[ci,12]),external_cached_center_depth=float(packet[oi,12]),
                        child_RGB=packet[ci,13:16].tolist(),external_RGB=packet[oi,13:16].tolist()))
        rows.append(dict(event=event,pixel=[x,y],active_children=len(children),active_crossing_pairs=len(crossings),pairs=crossings))
    write_new(target,dict(exact_centroid_and_swap_checks=True,sympy=s.__version__,rows=rows,
        trace_audit_sha256=sha(OUT/'independent_audit.json'),script_sha256=sha(Path(__file__)),
        posthoc_mechanism=True,physical_correct_order_proved=False,novelty_established=False,training=False))
    print(json.dumps(dict(exact_centroid_and_swap_checks=True,rows=[{k:v for k,v in r.items() if k!='pairs'} for r in rows])),flush=True)

if __name__=='__main__':main()
