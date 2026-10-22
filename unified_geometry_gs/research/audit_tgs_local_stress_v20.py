"""Independent selection and lossless output reconstruction audit, including hard cases."""
import json
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new
OUT=ROOT/'experiments/tgs_local_stress_v20'

def main():
    plan=json.loads((OUT/'plan.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert result['all_gates_passed'] and result['full_forwards']==26 and len(result['rows'])==16
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    selections=[];rows=[]
    for scene in plan['scenes']:
        folder=OUT/scene;s=json.loads((folder/'selection.json').read_text());eligible=np.load(folder/'eligible.npz');assert sha(folder/'eligible.npz')==s['eligible_sha256']
        ids=eligible['ids'];area=eligible['area'];assert len(set(ids.tolist()))==len(ids) and np.all(eligible['opacity']>=.01)
        assert np.array_equal(area,np.prod(eligible['hi'].astype(np.int64)-eligible['lo'].astype(np.int64),axis=1))
        chosen=[];amax=int(max(area));assert amax==s['max_parent_tiles']
        for fraction,en in zip([1,.5,.1,.01],s['parents']):
            ix=min((i for i in range(len(ids)) if int(ids[i]) not in chosen),key=lambda i:(abs(float(area[i])-amax*fraction),int(ids[i])))
            assert int(ids[ix])==en['parent_id'] and int(area[ix])==en['parent_tiles'];chosen.append(int(ids[ix]))
        baseline=np.load(folder/'baseline.npz');selections.append(dict(scene=scene,ids=chosen,max_parent_tiles=amax,selection_verified=True))
        for r in [x for x in result['rows'] if x['scene']==scene]:
            pf=folder/f"parent_{r['parent_index']}"/r['appearance'];patch=np.load(pf/'query_patch.npz');x0,y0,x1,y1=patch['bbox'].tolist()
            assert [x0,y0,x1,y1]==r['bbox']
            for k in ['rgb','alpha','paths']:
                a=baseline[k].copy()
                if k=='alpha':a[y0:y1,x0:x1]=patch[k]
                else:a[:,y0:y1,x0:x1]=patch[k]
                assert ahash(a)==r['query_hashes'][k]==r['native_hashes'][k],(scene,r['parent_index'],r['appearance'],k)
            assert all(r['checks'].values()) and r['all_defined_cache_fields_literal'] and r['all_tile_lists_literal'] and not any(r['tile_audit'][:4])
            assert 0<=r['candidate_pixel_visits_upper']<=r['all_candidate_pixel_visits_upper']
            rows.append(dict(scene=scene,parent_index=r['parent_index'],appearance=r['appearance'],coverage=r['affected_pixels']/r['pixels'],
                work_fraction=r['candidate_pixel_visits_upper']/r['all_candidate_pixel_visits_upper'],child_color_delta_max=r['child_color_delta_max'],reconstructed_all_three=True))
    write_new(OUT/'independent_audit.json',dict(all_passed=True,selections=selections,rows=rows,cases=16,training=False,
        max_coverage=max(x['coverage'] for x in rows),min_coverage=min(x['coverage'] for x in rows),
        scope='Selection and query outputs reconstructed independently; cache/list equality are recorded CUDA runtime gates'))
    print(json.dumps(dict(cases=16,max_coverage=max(x['coverage'] for x in rows),min_coverage=min(x['coverage'] for x in rows))),flush=True)

if __name__=='__main__':main()
