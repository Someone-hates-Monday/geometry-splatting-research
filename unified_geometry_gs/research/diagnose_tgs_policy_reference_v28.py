"""Post-failure, fixed first room reference probe; no policy/GT inspection."""
import gc,json,traceback
import numpy as np
import torch
from prepare_tgs_policy_data_v28 import ROOT,OUT,sha,write
from tgs_split_color_common_v14 import HELPER,module,ahash,pack,split,colors,CrossWorkload,exact
from tgs_aux_ablation_common_v21 import control
from tgs_batch_score_common_v22 import bank,backend as planner_backend
from tgs_packed_query_common_v25 import backend as emitter_backend,dynamic_layout,emit
from run_tgs_policy_transfer_v28 import camera,view_rows
from run_tgs_local_split_v19 import cache
from trace_tgs_window_scene_v15 import call
D=OUT/'reference_mismatch_probe'
def diff(a,b):
    if isinstance(a,torch.Tensor):a=a.detach().cpu().numpy()
    if isinstance(b,torch.Tensor):b=b.detach().cpu().numpy()
    assert a.shape==b.shape and a.dtype==b.dtype
    changed=a.view(np.uint32)!=b.view(np.uint32)
    return dict(literal=exact(a,b),different_bit_values=int(changed.sum()),different_numeric_values=int(np.count_nonzero(a!=b)),max_abs=float(np.abs(a.astype(float)-b.astype(float)).max(initial=0)))
@torch.no_grad()
def main():
    D.mkdir();choice=json.loads((OUT/'choices.json').read_text());scene='room';view='calibration_0';parent=min(next(s for s in choice['scenes'] if s['scene']==scene)['selected'].values())
    assert parent==42674 and not (OUT.parent.parent/'sources/tgs_policy_transfer_v28/heldout_manifest.json').exists()
    deps=[__file__,str(OUT/'choices.json'),str(OUT/'execution_plan.json'),str(OUT/'reference_calibration_failure.txt'),str(OUT/scene/'candidate_pool.json'),str(OUT/scene/view/'scores.json'),str(OUT/scene/view/'baseline_rgb.npy')]
    write(D/'plan.json',dict(case=dict(scene=scene,view=view,parent=parent),selection='first failing full-reference case from preserved traceback, not GT',inputs={str(__import__('pathlib').Path(p).relative_to(ROOT)):sha(p) for p in deps},native_forward_cap=4,new_GT_inspection=False,training=False,original_attempt_failed=True))
    ctrl=control();helper=module(HELPER);planner=planner_backend();emitter=emitter_backend();w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh);row=view_rows(scene,'calibration')[0];cam=camera(row);col=colors(helper,state,cam)
    original=call(ctrl,state,cam,col);torch.cuda.synchronize();calls=1;bg,br,bl,_=cache(original,state,cam);B=original[1].cpu().numpy();assert exact(B,np.load(OUT/scene/view/'baseline_rgb.npy'))
    pool=json.loads((OUT/scene/'candidate_pool.json').read_text())['rows'];idx=next(i for i,r in enumerate(pool) if r['parent_id']==parent);bs,cc,parents,small,cc_list=bank(state,col,cam,pool,helper)
    tiny=call(ctrl,bs,cam,cc);torch.cuda.synchronize();calls+=1;cg,_,_,_=cache(tiny,bs,cam)
    g=[x[4*idx:4*idx+4].contiguous() for x in cg];boxes,offsets,d,rois=dynamic_layout(planner,bg,g,parents[idx:idx+1].contiguous(),cam['w'],cam['h'])
    patch=emit(emitter,bg,br,bl,col,g,cc_list[idx],parents[idx:idx+1].contiguous(),boxes,offsets,original[1],d);torch.cuda.synchronize();assert not any(patch[1].cpu().tolist());R=rois[0];x0,y0,x1,y1=R
    cached=B.copy();cached[:,y0:y1,x0:x1]=patch[0].cpu().numpy().reshape(3,y1-y0,x1-x0);ref=next(r for r in json.loads((OUT/scene/view/'scores.json').read_text())['rows'] if r['parent_id']==parent);assert ahash(cached)==ref['rgb_sha256']
    trial=split(state,parent);tc=colors(helper,trial,cam);inherited=torch.cat([col[:parent],col[parent+1:],cc_list[idx]]).contiguous();color_diff=diff(tc,inherited)
    full=call(ctrl,trial,cam,tc);torch.cuda.synchronize();calls+=1;F=full[1].cpu().numpy();fg,_,_,_=cache(full,trial,cam)
    visible=bg[6]>0;visible[parent]=False;old=torch.nonzero(visible).flatten();new=old-(old>parent).long();external=[];children=[]
    for k in range(9):
        external.append(diff(bg[k].index_select(0,old),fg[k].index_select(0,new)))
        active=g[6]>0;children.append(diff(g[k][active],fg[k][-4:][active]))
    bits=F.view(np.uint32)!=cached.view(np.uint32);where=np.any(bits,axis=0);yy,xx=np.nonzero(where);box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else [0,0,0,0]
    outside=where.copy();outside[y0:y1,x0:x1]=False
    np.save(D/'full_native_rgb.npy',F,allow_pickle=False);np.savez_compressed(D/'cached_patch.npz',bbox=np.array(R,dtype=np.int32),rgb=cached[:,y0:y1,x0:x1])
    extra=None
    if not color_diff['literal']:
        control_full=call(ctrl,trial,cam,inherited);torch.cuda.synchronize();calls+=1;extra=diff(control_full[1],torch.from_numpy(cached).cuda());np.save(D/'inherited_color_full_rgb.npy',control_full[1].cpu().numpy(),allow_pickle=False)
    result=dict(original_baseline_literal=True,original_cached_score_image_reproduced=True,color_recompute_vs_inherited=color_diff,external_geometry=external,child_geometry=children,
        full_vs_cached=diff(F,cached),different_pixels=int(where.sum()),different_pixels_outside_structural_ROI=int(outside.sum()),difference_bbox=box,structural_ROI=R,full_with_inherited_colors_vs_cached=extra,
        new_native_forwards=calls,model_inputs_unchanged=initial==w.input_hashes(),new_GT_inspection=False,training=False,heldout_unread=True)
    assert calls<=4 and initial==w.input_hashes();write(D/'results.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        if D.exists():
            with (D/'failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
