"""Prespecified descriptive transfer analysis; no policy retuning or significance."""
import json
from pathlib import Path
import numpy as np
from prepare_tgs_policy_data_v28 import ROOT,OUT,sha,write
POLICIES=['signed_mixed','signed_MSE','perturbation','largest_area','hash']
def main():
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_passed'];choices=json.loads((OUT/'choices.json').read_text());scenes=[];inputs=[Path(__file__),OUT/'choices.json',OUT/'independent_audit.json']
    for sc in choices['scenes']:
        scene=sc['scene'];tables={}
        for cohort in ['calibration','heldout']:
            for vi in range(2):
                view=f'{cohort}_{vi}';p=OUT/scene/view/'scores.json';tables[view]=json.loads(p.read_text())['rows'];inputs.append(p)
        pool=json.loads((OUT/scene/'candidate_pool.json').read_text())['rows'];pool_test=[]
        for parent in [r['parent_id'] for r in pool]:
            vals=[next(t for t in tables[f'heldout_{v}'] if t['parent_id']==parent)['mixed_L1_DSSIM'] for v in range(2)]
            pool_test.append(dict(parent_id=parent,mean_mixed=float(np.mean(vals)),per_view_mixed=vals))
        ranked=sorted(pool_test,key=lambda r:(r['mean_mixed'],r['parent_id']));policies=[]
        for name in POLICIES+['guarded_signed_mixed','no_edit']:
            parent=sc['selected'][name] if name in POLICIES else sc['guarded_signed_mixed'] if name=='guarded_signed_mixed' else None
            metrics={}
            for cohort in ['calibration','heldout']:
                refs=[next(r for r in audit['rows'] if (r['scene'],r['view'],r['parent_id'])==(scene,f'{cohort}_{vi}',parent)) for vi in range(2)] if parent is not None else []
                scores=[r['CPU_score'] for r in refs] if refs else [[0.,0.,0.],[0.,0.,0.]]
                mixes=[r['CPU_mixed'] for r in refs] if refs else [0.,0.]
                metrics[cohort]=dict(per_view_score=scores,mean_score=np.mean(scores,axis=0).tolist(),per_view_mixed=mixes,mean_mixed=float(np.mean(mixes)),per_view_raw_PSNR_change_dB=[r['raw_PSNR_change_dB'] for r in refs] if refs else [0.,0.],mean_raw_PSNR_change_dB=float(np.mean([r['raw_PSNR_change_dB'] for r in refs])) if refs else 0.)
            testmean=metrics['heldout']['mean_mixed'];calmean=metrics['calibration']['mean_mixed'];rank=next(i+1 for i,r in enumerate(ranked) if r['parent_id']==parent) if parent is not None else None
            policies.append(dict(policy=name,parent_id=parent,added_triangles=3 if parent is not None else 0,**metrics,calibration_improved_beyond_atol=calmean < -1e-13,heldout_improved_beyond_atol=testmean < -1e-13,heldout_worse_beyond_atol=testmean > 1e-13,heldout_pool_rank=rank,heldout_cached_pool_regret=testmean-ranked[0]['mean_mixed']))
        primary=policies[0];comparisons={r['policy']:dict(primary_minus_comparator_mean_mixed=primary['heldout']['mean_mixed']-r['heldout']['mean_mixed'],primary_lower_beyond_atol=primary['heldout']['mean_mixed']<r['heldout']['mean_mixed']-1e-13) for r in policies[1:5]}
        scenes.append(dict(scene=scene,policies=policies,primary_comparisons=comparisons,cached_test_pool=pool_test,cached_test_oracle=ranked[0],oracle_not_implementable=True,views_previously_unscored=True,independent_scene=False))
    write(OUT/'analysis.json',dict(all_verified=True,scenes=scenes,total_native_forwards=audit['total_new_native_forwards'],selected_full_references=len(audit['rows']),maximum_CPU_GPU_error=audit['maximum_selected_CPU_GPU_error'],inputs={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},new_native_forwards=0,training=False,final_trained_quality_gain=False,geometry_quality_measured=False,novelty_established=False,goal_complete=False))
    print(json.dumps(dict(stage='TRANSFER_ANALYSIS_DONE',scenes=[dict(scene=s['scene'],primary=s['policies'][0],comparisons=s['primary_comparisons']) for s in scenes])),flush=True)
if __name__=='__main__':main()
