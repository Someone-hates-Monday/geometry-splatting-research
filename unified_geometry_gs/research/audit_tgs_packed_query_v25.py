"""Independent CPU reconstruction of all32 packed outputs and sealed score identities."""
import json
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,exact
OUT=ROOT/'experiments/tgs_packed_query_v25';V20=ROOT/'experiments/tgs_local_stress_v20';V23=ROOT/'experiments/tgs_view_transfer_v23';V24=ROOT/'experiments/tgs_local_image_loss_v24'

def main():
    for name in ['plan.json','runtime_dependency_addendum.json']:
        for n,h in json.loads((OUT/name).read_text())['inputs'].items():assert sha(ROOT/n)==h,n
    r=json.loads((OUT/'results.json').read_text());assert r['all_passed'] and r['new_native_forwards']==8 and len(r['rows'])==32
    refs=json.loads((V23/'analysis.json').read_text());loss=json.loads((V24/'results.json').read_text());rows=[]
    for scene in ['flowers','room']:
        for view in ['fresh','old_0']:
            packed=np.load(OUT/f'{scene}_{view}/packed_rgb.npz');base=np.load(V20/scene/'baseline.npz')['rgb'] if view=='fresh' else np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy')
            rr=[x for x in r['rows'] if x['scene']==scene and x['view']==view]
            assert packed['offsets'][0]==0 and packed['offsets'][-1]==packed['rgb'].size
            for e,en in enumerate(rr):
                ref=next(x for x in refs['rows'] if (x['scene'],x['view'],x['event'])==(scene,view,e));score=next(x for x in loss['rows'] if (x['scene'],x['view'],x['event'])==(scene,view,e))
                x0,y0,x1,y1=packed['rois'][e].tolist();a,b=packed['offsets'][e:e+2];patch=packed['rgb'][int(a):int(b)].reshape(3,y1-y0,x1-x0)
                old=V20/scene/f"parent_{ref['parent_index']}"/ref['appearance']/'query_patch.npz' if view=='fresh' else V23/scene/f'proposal_{e}_rgb_patch.npz'
                data=np.load(old);assert data['bbox'].tolist()==[x0,y0,x1,y1]==en['ROI'] and exact(data['rgb'],patch) and ahash(patch)==en['patch_hash']
                full=base.copy();full[:,y0:y1,x0:x1]=patch;assert ahash(full)==en['full_rgb_hash']==ref['rgb_sha256']
                assert exact(packed['scores'][e],np.array(score['score'],dtype=np.float64)) and en['score']==score['score']
                rows.append(dict(scene=scene,view=view,event=e,RGB_and_score_literal=True,full_rgb_hash=ahash(full)))
    write_new(OUT/'independent_audit.json',dict(all_passed=True,rows=rows,new_native_forwards=0,training=False))
    print('ALL32_PACKED_OUTPUTS_CPU_AUDITED',flush=True)

if __name__=='__main__':main()
