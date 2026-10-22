"""Readonly v14 patch reconstruction verifies all recorded local full-image RGB hashes."""
import json
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new
OUT=ROOT/'experiments/tgs_local_split_v19';OLD=ROOT/'experiments/tgs_split_color_v14'

def main():
    p=json.loads((OUT/'plan.json').read_text());r=json.loads((OUT/'results.json').read_text());assert r['all_gates_passed'] and r['full_forwards']==68
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    assert len(r['rows'])==32 and {(x['scene'],x['camera'],x['event']) for x in r['rows']}=={(x['scene'],x['camera'],x['event']) for x in p['cases']}
    baselines={};rows=[]
    for en in r['rows']:
        scene,ci,e=en['scene'],en['camera'],en['event'];key=(scene,ci)
        if key not in baselines:baselines[key]=np.load(OLD/scene/f'baseline_{ci}/rgb.npy')
        patch=np.load(OLD/scene/f'event_{e}'/f'camera_{ci}/changes.npz');x0,y0,x1,y1=patch['bbox'].tolist()
        a=baselines[key].copy();a[:,y0:y1,x0:x1]=patch['hold_rgb'];h=ahash(a)
        assert h==en['query_RGB_sha256']==en['native_RGB_sha256']
        assert all(en['checks'].values()) and en['max_RGB_error']==0 and en['all_defined_cache_fields_literal'] and en['all_tile_lists_literal']
        assert not any(en['tile_audit'][:4]) and en['affected_pixels']<=en['pixels']
        rows.append(dict(scene=scene,camera=ci,event=e,RGB_reference_reconstructed=True,
            affected_fraction=en['affected_pixels']/en['pixels'],tile_records_verified=en['tile_audit'][4]))
    write_new(OUT/'independent_audit.json',dict(all_passed=True,rows=rows,zero_affected=sum(x['affected_tiles']==0 for x in r['rows']),
        max_affected_fraction=max(x['affected_fraction'] for x in rows),tile_records_verified=sum(x['tile_records_verified'] for x in rows),
        scope='Independent reconstruction of reference hashes; global kernel/cache/list gates are recorded runtime checks',training=False))
    print(json.dumps(dict(all_passed=True,cases=len(rows),zero_affected=sum(x['affected_tiles']==0 for x in r['rows']),max_affected_fraction=max(x['affected_fraction'] for x in rows))),flush=True)

if __name__=='__main__':main()
