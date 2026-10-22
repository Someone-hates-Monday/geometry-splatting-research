"""Posthoc coverage description after transfer analysis, no new candidate selection."""
import json
from pathlib import Path
from prepare_tgs_policy_data_v28 import ROOT,OUT,sha,write
def main():
    inputs=[Path(__file__),OUT/'analysis.json',OUT/'choices.json'];choices=json.loads((OUT/'choices.json').read_text());tables={}
    for sc in choices['scenes']:
        for cohort in ['calibration','heldout']:
            for vi in range(2):
                p=OUT/sc['scene']/f'{cohort}_{vi}'/'scores.json';inputs.append(p);tables[(sc['scene'],cohort,vi)]=json.loads(p.read_text())
    write(OUT/'coverage_plan.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},posthoc_after_transfer_aggregate=True,new_native_forwards=0,changes_selection=False,observed_RGB_support_not_general_guarantee=True))
    scenes=[]
    for sc in choices['scenes']:
        scene=sc['scene'];pool=json.loads((OUT/scene/'candidate_pool.json').read_text())['rows'];rows=[];matrix=[[0]*3 for _ in range(3)]
        for p in pool:
            summaries={}
            for cohort in ['calibration','heldout']:
                per=[]
                for vi in range(2):
                    tab=tables[(scene,cohort,vi)];r=next(t for t in tab['rows'] if t['parent_id']==p['parent_id']);x0,y0,x1,y1=r['ROI'];per.append(dict(view=f'{cohort}_{vi}',structural_ROI_empty=x0==x1 or y0==y1,RGB_changed=r['rgb_sha256']!=tab['baseline_RGB_sha256'],ROI=r['ROI']))
                summaries[cohort]=dict(views=per,RGB_changed_views=sum(r['RGB_changed'] for r in per),nonempty_structural_views=sum(not r['structural_ROI_empty'] for r in per))
            matrix[summaries['calibration']['RGB_changed_views']][summaries['heldout']['RGB_changed_views']]+=1
            rows.append(dict(parent_id=p['parent_id'],area_bin=p['area_bin'],**summaries))
        primary=next(r for r in rows if r['parent_id']==sc['selected']['signed_mixed']);scenes.append(dict(scene=scene,rows=rows,RGB_changed_view_count_matrix_calibration_by_heldout=matrix,primary=primary))
    result=dict(scenes=scenes,plan_sha256=sha(OUT/'coverage_plan.json'),posthoc_descriptive=True,causal_explanation_established=False,selection_unchanged=True,new_native_forwards=0,training=False)
    write(OUT/'coverage_results.json',result);print(json.dumps(dict(stage='POSTHOC_COVERAGE_DONE',scenes=[dict(scene=s['scene'],matrix=s['RGB_changed_view_count_matrix_calibration_by_heldout'],primary=s['primary']) for s in scenes])),flush=True)
if __name__=='__main__':main()
