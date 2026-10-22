"""Equal-budget descriptive results, with no retuning and no inferred training gains."""
import json
from pathlib import Path
import numpy as np
from tgs_joint_policy_data_v29 import ROOT,OUT,sha,write
POLICIES=['signed_mixed','signed_MSE','perturbation','largest_min_area','hash']
def main():
    a=json.loads((OUT/'independent_audit.json').read_text());assert a['all_passed'];choices=json.loads((OUT/'choices.json').read_text());reg=json.loads((OUT/'global_reg_results.json').read_text());assert reg['all_identities_passed'];scenes=[]
    for sc in choices['scenes']:
        policies=[];rscene=next(r for r in reg['scenes'] if r['scene']==sc['scene'])
        for name in POLICIES+['guarded_signed_mixed','no_edit']:
            is_no_edit=name=='no_edit' or name=='guarded_signed_mixed' and not sc['guarded_signed_mixed'];source='signed_mixed' if name=='guarded_signed_mixed' else name;stats={}
            coefficient=0. if is_no_edit else next(r['joint_coefficient'] for r in rscene['rows'] if r['policy']==source)
            for cohort in ['calibration','heldout']:
                rs=[next(r for r in a['rows'] if r['scene']==sc['scene'] and r['view']==f'{cohort}_{v}' and source in r['policies']) for v in range(2)] if not is_no_edit else []
                s=[r['CPU_score'] for r in rs] if rs else [[0.,0.,0.],[0.,0.,0.]];mix=[r['CPU_mixed'] for r in rs] if rs else [0.,0.];psnr=[r['raw_PSNR_change_dB'] for r in rs] if rs else [0.,0.];mean=float(np.mean(mix))
                stats[cohort]=dict(per_view_score=s,mean_score=np.mean(s,axis=0).tolist(),per_view_mixed=mix,mean_mixed=mean,per_view_raw_PSNR_change_dB=psnr,mean_raw_PSNR_change_dB=float(np.mean(psnr)),per_view_RGB_changed_pixels=[r['RGB_changed_pixels'] for r in rs] if rs else [0,0],illustrative_image_plus_mean_opacity_break_even_lambda=float(-mean/coefficient) if coefficient and mean*coefficient<0 else None)
            policies.append(dict(policy=name,parents=[] if is_no_edit else sc['selected'][source],added_triangles=0 if is_no_edit else 24,exact_value_opacity_mean_coefficient=coefficient,coefficient_excludes_unknown_lambda=True,**stats))
        primary=policies[0];comparisons={p['policy']:dict(primary_minus_comparator_heldout_mean_mixed=primary['heldout']['mean_mixed']-p['heldout']['mean_mixed'],primary_lower_beyond_atol=primary['heldout']['mean_mixed'] < p['heldout']['mean_mixed']-1e-13) for p in policies[1:5]}
        scenes.append(dict(scene=sc['scene'],eligible_count=json.loads((OUT/sc['scene']/'candidate_pool.json').read_text())['eligible_count'],policies=policies,primary_comparisons=comparisons,independent_scene=False))
    deps=[Path(__file__),OUT/'choices.json',OUT/'independent_audit.json',OUT/'global_reg_results.json']
    write(OUT/'analysis.json',dict(all_verified=True,scenes=scenes,native_forwards=a['total_native_forwards'],joint_references=a['audited_joint_references'],maximum_CPU_GPU_error=a['maximum_CPU_GPU_score_error'],maximum_calibration_additivity_error=a['maximum_calibration_unary_sum_CPU_error'],inputs={p.relative_to(ROOT).as_posix():sha(p) for p in deps},supplemental_reg_did_not_change_choices=True,full_training_objective_measured=False,training=False,geometry_measured=False,final_quality_or_novelty_established=False,goal_complete=False))
    print(json.dumps(dict(stage='JOINT_TRANSFER_ANALYSIS_DONE',scenes=[dict(scene=s['scene'],primary=s['policies'][0],comparisons=s['primary_comparisons']) for s in scenes])),flush=True)
if __name__=='__main__':main()
