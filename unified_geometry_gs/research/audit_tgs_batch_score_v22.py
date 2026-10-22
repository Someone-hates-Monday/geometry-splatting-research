"""CPU-only reconstruction of every reference image, bank input and integer score."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,exact
from tgs_image_metrics_v11 import png8
from tgs_batch_score_common_v22 import OUT,cpu_sse
V20=ROOT/'experiments/tgs_local_stress_v20'

def main():
    p=json.loads((OUT/'plan.json').read_text());r=json.loads((OUT/'results.json').read_text());assert r['all_gates_passed'] and r['full_forwards']==20 and len(r['rows'])==16
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    rows=[]
    for scene in ['flowers','room']:
        base=np.load(V20/scene/'baseline.npz');gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy')
        for name in ['original.JPG','published_gt.png']:assert np.array_equal(np.asarray(Image.open(ROOT/f'sources/tgs_sh_isolation_v12/{scene}/fresh'/name).convert('RGB')),gt)
        baseline_sse=cpu_sse(png8(base['rgb']),gt);bank=np.load(OUT/scene/'bank_inputs.npz')
        for e,en in enumerate([x for x in r['rows'] if x['scene']==scene]):
            assert en['event']==e and int(bank['parents'][e])==en['parent_id']
            old=next(x for x in p['cases'] if x['scene']==scene and x['parent_index']==en['parent_index'] and x['appearance']==en['appearance'])
            folder=V20/scene/f"parent_{en['parent_index']}";params=np.load(folder/'input.npz');patch=np.load(folder/en['appearance']/'query_patch.npz');x0,y0,x1,y1=patch['bbox'].tolist()
            for key,original in [('points','child_points'),('sigma','sigma'),('opacity','opacity')]:assert exact(bank[key][4*e:4*e+4],params[original])
            assert exact(bank['colors'][4*e:4*e+4],params['hold_colors' if en['appearance']=='hold' else 'moved_colors'])
            images={}
            for key in ['rgb','alpha','paths']:
                a=base[key].copy()
                if key=='alpha':a[y0:y1,x0:x1]=patch[key]
                else:a[:,y0:y1,x0:x1]=patch[key]
                assert ahash(a)==old['native_hashes'][key]==en['debug_hashes'][key];images[key]=a
            delta=cpu_sse(png8(images['rgb']),gt)-baseline_sse
            assert delta==en['delta_SSE']==en['batch_score']==en['full_gpu_score'] and baseline_sse==en['baseline_SSE']
            assert all(en['checks'].values()) and en['all_tile_lists_literal'] and en['visible_cache_fields_literal']
            assert baseline_sse+delta>=0
            rows.append(dict(scene=scene,event=e,delta_SSE=delta,exact=True,PNG8_MSE_delta=delta/(gt.size*255**2)))
    write_new(OUT/'independent_audit.json',dict(all_passed=True,rows=rows,negative_scores=sum(x['delta_SSE']<0 for x in rows),positive_scores=sum(x['delta_SSE']>0 for x in rows),
        zero_scores=sum(x['delta_SSE']==0 for x in rows),training=False,score_validation_not_policy_evaluation=True,source_sha256=sha(Path(__file__))))
    print(json.dumps(dict(all_passed=True,scores=len(rows),negative=sum(x['delta_SSE']<0 for x in rows),positive=sum(x['delta_SSE']>0 for x in rows),zero=sum(x['delta_SSE']==0 for x in rows))),flush=True)

if __name__=='__main__':main()
