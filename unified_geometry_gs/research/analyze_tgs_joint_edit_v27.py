"""Summarize the aggregate-null result without manufacturing an interaction finding."""
import json,math
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_joint_edit_common_v27 import OUT

def main():
    r=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text());post=json.loads((OUT/'support_diagnostic_results.json').read_text());reg=json.loads((OUT/'global_reg_results.json').read_text())
    assert r['all_passed'] and len(r['rows'])==24 and audit['all_passed'] and reg['all_identities_passed'] and len(post['rows'])==2
    max_eta=[max(abs(x['interactions'][k]) for x in r['rows']) for k in range(3)]
    all_within=all(abs(x['interactions'][k])<=x['roundoff_bounds'][k] for x in r['rows'] for k in range(3))
    assert all_within and all(x['PNG8_interaction']==0 for x in r['rows'])
    for x in post['rows']:
        assert x['observed_RGB_change_overlap_pixels']==0 and x['actual_joint_equals_observed_piecewise_mosaic'] and x['computed_RGB_interaction_peak']==0 and x['computed_CPU_SSIM_interaction_peak']==0
    view_rows=[]
    for scene in ['flowers','room']:
        for view in ['fresh','old_0']:
            rows=[x for x in r['rows'] if x['scene']==scene and x['view']==view];assert len(rows)==6
            view_rows.append(dict(scene=scene,view=view,pairs=6,structural_ssim_disjoint=sum(x['classification']['ssim_disjoint'] for x in rows),
                all_PNG8_interactions_zero=all(x['PNG8_interaction']==0 for x in rows),
                maximum_blend_interactions=[max(abs(x['blend_interactions'][k]) for x in rows) for k in range(2)],
                unary_sum_joint_blend_sign_disagreements=sum(math.copysign(1,x['unary_blend_sum'][k])!=math.copysign(1,x['joint_blends'][k]) for x in rows for k in range(2))))
    out=dict(all_verified=True,new_native_forwards=32,joint_cases=24,views=view_rows,metrics=['raw_MSE','raw_L1','raw_SSIM'],maximum_GPU_interaction_abs=max_eta,
        maximum_CPU_GPU_joint_score_error=audit['max_CPU_GPU_joint_score_error'],all_aggregate_residuals_within_roundoff_reference=all_within,
        structural_disjoint_pairs=22,nontrivial_structural_disjoint_pairs=audit['nontrivial_ssim_disjoint_pairs'],overlapping_box_pairs=2,
        overlapping_box_actual_RGB_and_CPU_SSIM_effects_separated=True,substantive_real_interaction_demonstrated=False,
        global_mean_opacity_interaction_symbolically_nonadditive=True,full_training_objective_measured=False,
        training=False,accepted_mutations=0,independent_scene_or_quality_gain=False,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'results.json',OUT/'independent_audit.json',OUT/'support_diagnostic_results.json',OUT/'global_reg_results.json',Path(__file__)]})
    write_new(OUT/'analysis.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':main()
