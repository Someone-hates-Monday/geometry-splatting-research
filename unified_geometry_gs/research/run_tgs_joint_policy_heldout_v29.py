"""Heldout entry accepts the already-created identity-audit GT directories."""
import gc,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_joint_policy_data_v29 import ROOT,OUT,SOURCE,sha,write
from run_tgs_joint_policy_v29 import frozen,rows,joint_refs
from tgs_split_color_common_v14 import HELPER,module,pack,colors,CrossWorkload
from tgs_aux_ablation_common_v21 import control
from tgs_shared_image_loss_common_v26 import backend as loss_backend
from tgs_local_image_loss_common_v24 import layout,device_layout
from run_tgs_real_events import camera_tensors
from trace_tgs_window_scene_v15 import call
@torch.no_grad()
def main():
    frozen();assert json.loads((OUT/'heldout_identity.json').read_text())['all_passed'];choices=json.loads((OUT/'choices.json').read_text());assert json.loads((SOURCE/'manifest.json').read_text())['choices_sha256']==sha(OUT/'choices.json')
    deps=[Path(__file__),ROOT/'run_tgs_joint_policy_v29.py',OUT/'execution_plan.json',OUT/'choices.json',OUT/'heldout_identity.json',SOURCE/'manifest.json']
    write(OUT/'heldout_execution_addendum.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in deps},reason='source review before execution: original baseline helper assumes new directory, but identity stage creates GT directory; use same full native baseline without that mkdir',previous_heldout_entry_executed=False,view_or_policy_changes=False,native_forward_cap=24,training=False))
    ctrl=control();helper=module(HELPER);losser=loss_backend();calls=0
    for sc in choices['scenes']:
        scene=sc['scene'];w=CrossWorkload(scene);initial=w.input_hashes();assert initial==sc['model_inputs'];state=pack(w.points,w.sigma,w.opacity,w.sh)
        for row in rows(scene,'heldout'):
            H,W={'flowers':(828,1256),'room':(1038,1557)}[scene];cam=camera_tensors(row['camera'],W,H);col=colors(helper,state,cam);native=call(ctrl,state,cam,col);torch.cuda.synchronize();calls+=1;assert torch.isfinite(native[1]).all();B=native[1].cpu().numpy();dest=OUT/scene/row['view'];assert dest.is_dir() and not (dest/'baseline_rgb.npy').exists();np.save(dest/'baseline_rgb.npy',B,allow_pickle=False)
            gt=np.load(ROOT/row['GT']);assert gt.shape==(H,W,3);gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda();fl=device_layout(layout([[0,0,W,H]],W,H));q=dict(row=row,cam=cam,col=col,native=native,base=native[1],B=B,dest=dest,gt8=gt8,fl=fl,shared=None)
            calls+=joint_refs(ctrl,helper,losser,state,q,sc);del q,native,B,col,gt8,fl;gc.collect();torch.cuda.empty_cache()
        assert initial==w.input_hashes();del w,state;gc.collect();torch.cuda.empty_cache()
    total=calls+json.loads((OUT/'calibration_results.json').read_text())['new_native_forwards'];assert calls<=24 and total<=52;write(OUT/'heldout_results.json',dict(all_passed=True,new_native_forwards=calls,total_native_forwards=total,choices_sha256=sha(OUT/'choices.json'),heldout_entry_addendum_sha256=sha(OUT/'heldout_execution_addendum.json'),model_inputs_unchanged=True,training=False));print(json.dumps(dict(stage='ALL_NEW_HELDOUT_JOINT_REFERENCES_PASSED',native_total=total)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'heldout_execution_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
