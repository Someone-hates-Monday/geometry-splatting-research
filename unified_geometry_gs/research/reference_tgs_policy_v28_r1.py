"""Explicit fixed-external-FP32-color references, recovering preserved partial work."""
import argparse,gc,json,traceback
import numpy as np
import torch
from prepare_tgs_policy_data_v28 import ROOT,OUT,sha,write
from run_tgs_policy_transfer_v28 import frozen,view_rows,camera
from tgs_split_color_common_v14 import HELPER,module,ahash,pack,split,colors,CrossWorkload,exact
from tgs_batch_score_common_v22 import child_only
from tgs_aux_ablation_common_v21 import control
from trace_tgs_window_scene_v15 import call
ADD=ROOT/'TGS_POLICY_REFERENCE_ADDENDUM_V28_R1.md'

@torch.no_grad()
def main(cohort):
    frozen();D=OUT/'reference_mismatch_probe';probe=json.loads((D/'results.json').read_text());assert probe['full_with_inherited_colors_vs_cached']['literal'] and probe['new_native_forwards']==4
    assert not (OUT/(cohort+'_reference_receipt.json')).exists();choices=json.loads((OUT/'choices.json').read_text())
    deps=[__file__,ADD,OUT/'choices.json',OUT/'execution_plan.json',OUT/'reference_calibration_failure.txt',D/'plan.json',D/'results.json',D/'inherited_color_full_rgb.npy']
    write(OUT/(cohort+'_reference_r1_plan.json'),dict(inputs={__import__('pathlib').Path(p).relative_to(ROOT).as_posix():sha(p) for p in deps},cohort=cohort,original_full_recompute_gate_failed=True,external_FP32_colors_fixed=True,calibration_scores_reused_without_retuning=True,training=False))
    if cohort=='heldout':assert json.loads((OUT/'heldout_receipt.json').read_text())['all_passed']
    else:assert not (ROOT/'sources/tgs_policy_transfer_v28/heldout_manifest.json').exists()
    ctrl=control();helper=module(HELPER);rows=[];calls=0;reused=0
    for scene in ['flowers','room']:
        sc=next(s for s in choices['scenes'] if s['scene']==scene);w=CrossWorkload(scene);initial=w.input_hashes();assert initial==sc['model_inputs'];state=pack(w.points,w.sigma,w.opacity,w.sh)
        for row in view_rows(scene,cohort):
            view=row['view'];dest=OUT/scene/view;B=np.load(dest/'baseline_rgb.npy');table=json.loads((dest/'scores.json').read_text());assert ahash(B)==table['baseline_RGB_sha256'];cam=camera(row);col=colors(helper,state,cam)
            for parent in sorted(set(sc['selected'].values())):
                ref=next(t for t in table['rows'] if t['parent_id']==parent);x0,y0,x1,y1=ref['ROI'];target=dest/f'parent_{parent}_full_reference.npz';origin='new_corrected_native_reference'
                if cohort=='calibration' and scene=='flowers':
                    data=np.load(target);rgb=B.copy();rgb[:,y0:y1,x0:x1]=data['rgb'];assert data['bbox'].tolist()==ref['ROI'];reused+=1;origin='preserved_original_passed_reference'
                elif cohort=='calibration' and scene=='room' and view=='calibration_0' and parent==42674:
                    rgb=np.load(D/'inherited_color_full_rgb.npy');assert not target.exists();np.savez_compressed(target,bbox=np.array(ref['ROI'],dtype=np.int32),rgb=rgb[:,y0:y1,x0:x1]);reused+=1;origin='saved_fixed_probe_full_native_reference'
                else:
                    trial=split(state,parent);child=child_only(state,parent);cc=colors(helper,child,cam);tc=torch.cat([col[:parent],col[parent+1:],cc]).contiguous()
                    out=call(ctrl,trial,cam,tc);torch.cuda.synchronize();calls+=1;rgb=out[1].cpu().numpy();assert ahash(rgb)==ref['rgb_sha256'],(scene,view,parent)
                    assert not target.exists();np.savez_compressed(target,bbox=np.array(ref['ROI'],dtype=np.int32),rgb=rgb[:,y0:y1,x0:x1]);del trial,child,cc,tc,out
                assert ahash(rgb)==ref['rgb_sha256'];rebuilt=B.copy();rebuilt[:,y0:y1,x0:x1]=rgb[:,y0:y1,x0:x1];assert exact(rebuilt,rgb)
                rows.append(dict(scene=scene,view=view,parent_id=parent,full_RGB_literal_cached=True,full_rgb_sha256=ahash(rgb),patch_sha256=sha(target),ROI_outside_unchanged=True,origin=origin))
                print(json.dumps(dict(stage='R1_REFERENCE_PASSED',scene=scene,view=view,parent=parent,origin=origin,new_r1_calls=calls)),flush=True);del rgb,rebuilt;gc.collect()
        assert initial==w.input_hashes();del w,state;gc.collect();torch.cuda.empty_cache()
    if cohort=='calibration':assert calls==7 and reused==11 and len(rows)==18
    else:assert calls==18 and reused==0 and len(rows)==18
    write(OUT/(cohort+'_reference_receipt.json'),dict(all_passed=True,rows=rows,choices_sha256=sha(OUT/'choices.json'),new_native_forwards=calls+(11+4 if cohort=='calibration' else 0),new_r1_calls=calls,reused_full_native_references=reused,original_attempt_exit=1,original_attempt_native_calls=11,diagnostic_native_calls=4,prior_failed_calls_counted=cohort=='calibration',numerical_contract='fixed_external_FP32_colors_child_only_SH',r1_plan_sha256=sha(OUT/(cohort+'_reference_r1_plan.json')),training=False,model_inputs_unchanged=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cohort',choices=['calibration','heldout']);a=p.parse_args()
    try:main(a.cohort)
    except BaseException:
        with (OUT/(a.cohort+'_reference_r1_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
