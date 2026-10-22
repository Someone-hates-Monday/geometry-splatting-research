"""Post-hoc CPU diagnostic: zero aggregate interaction could hide map cancellation."""
import json
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,exact
from tgs_image_metrics_v11 import ssim_map
from tgs_joint_edit_common_v27 import OUT
V20=ROOT/'experiments/tgs_local_stress_v20';V23=ROOT/'experiments/tgs_view_transfer_v23'

def main():
    result=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_passed']
    cases=[r for r in result['rows'] if not r['classification']['ssim_disjoint']];assert len(cases)==2
    write_new(OUT/'support_diagnostic_plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'plan.json',OUT/'results.json',OUT/'independent_audit.json',Path(__file__)]},
        data_informed_post_hoc=True,purpose='Distinguish actual separation from per-pixel cancellation in the two overlapping-box aggregate-null cases',
        selected_cases=[dict(scene=r['scene'],view=r['view'],i=r['i'],j=r['j']) for r in cases],new_native_forwards=0,training=False))
    plan=json.loads((OUT/'plan.json').read_text());refs=json.loads((V23/'analysis.json').read_text());rows=[]
    for row in cases:
        scene,view=row['scene'],row['view'];vp=next(v for v in plan['views'] if (v['scene'],v['view'])==(scene,view));B=np.load(V20/scene/'baseline.npz')['rgb'] if view=='fresh' else np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy')
        gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy' if view=='fresh' else ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy');G=gt.astype(np.float64)/255.;b=B.transpose(1,2,0).astype(np.float64);sb=ssim_map(b,G);singles=[]
        for k in [row['i'],row['j']]:
            c=vp['candidates'][k];ref=next(r for r in refs['rows'] if (r['scene'],r['view'],r['event'])==(scene,view,c['event']));path=V20/scene/f"parent_{ref['parent_index']}"/'moved/query_patch.npz' if view=='fresh' else V23/scene/f"proposal_{ref['event']}_rgb_patch.npz"
            p=np.load(path);x0,y0,x1,y1=p['bbox'].tolist();rgb=B.copy();rgb[:,y0:y1,x0:x1]=p['rgb'];assert ahash(rgb)==ref['rgb_sha256'];v=rgb.transpose(1,2,0).astype(np.float64)
            singles.append(dict(rgb=rgb,d=v-b,ds=ssim_map(v,G)-sb))
        p=np.load(OUT/f"{scene}_{view}/pair_{row['i']}_{row['j']}_rgb_patch.npz");x0,y0,x1,y1=p['bbox'].tolist();joint=B.copy();joint[:,y0:y1,x0:x1]=p['rgb'];assert ahash(joint)==row['full_rgb_hash'];v=joint.transpose(1,2,0).astype(np.float64)
        observed=[np.any(s['rgb']!=B,axis=0) for s in singles];mosaic=B.copy()
        for s,mask in zip(singles,observed):mosaic[:,mask]=s['rgb'][:,mask]
        rgb_interaction=(v-b)-singles[0]['d']-singles[1]['d'];ssim_interaction=(ssim_map(v,G)-sb)-singles[0]['ds']-singles[1]['ds']
        out=dict(scene=scene,view=view,i=row['i'],j=row['j'],observed_RGB_change_overlap_pixels=int((observed[0]&observed[1]).sum()),
            actual_joint_equals_observed_piecewise_mosaic=exact(joint,mosaic),computed_RGB_interaction_peak=float(np.abs(rgb_interaction).max()),
            computed_RGB_interaction_nonzero_channels=int(np.count_nonzero(rgb_interaction)),
            observed_CPU_SSIM_change_overlap_channels=int(((singles[0]['ds']!=0)&(singles[1]['ds']!=0)).sum()),
            computed_CPU_SSIM_interaction_peak=float(np.abs(ssim_interaction).max()),computed_CPU_SSIM_interaction_nonzero_channels=int(np.count_nonzero(ssim_interaction)))
        rows.append(out);print(json.dumps(out),flush=True)
    write_new(OUT/'support_diagnostic_results.json',dict(rows=rows,data_informed_post_hoc=True,CPU_maps_not_new_GPU_maps=True,new_native_forwards=0,training=False,
        observed_support_not_a_structural_guarantee=True,plan_sha256=sha(OUT/'support_diagnostic_plan.json')))

if __name__=='__main__':main()
