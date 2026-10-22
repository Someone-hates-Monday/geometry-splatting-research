"""Separate one-shot setup measurement; not a reconstruction of lost run metadata."""
import gc,json,time
from pathlib import Path
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,write_new,module,pack,colors,CrossWorkload
from tgs_aux_ablation_common_v21 import control,view_list
from tgs_batch_score_common_v22 import OUT,backend
from run_tgs_local_split_v19 import cache
from trace_tgs_window_scene_v15 import call
from benchmark_tgs_local_split_v19 import storage_bytes,telemetry
TIM=OUT/'timing'

@torch.no_grad()
def main():
    r=json.loads((TIM/'results_recovered.json').read_text());assert r['full_forwards']==982
    files=[Path(__file__),TIM/'results_recovered.json',TIM/'recovery_plan.json',OUT/'plan.json',OUT/'build_receipt.json',ROOT/'benchmark_tgs_local_split_v19.py']
    write_new(TIM/'setup_addendum_plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},
        two_new_forwards=True,no_repeats=True,separate_process_from_main_timing=True,not_reconstruction_of_original_setup=True,training=False))
    ctrl=control();m=backend();helper=module(HELPER);rows=[];before=telemetry()
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);state=pack(w.points,w.sigma,w.opacity,w.sh);cam=next(x['cam'] for x in view_list(w,scene) if x['label']=='fresh');col=colors(helper,state,cam)
        torch.cuda.synchronize();t0=time.perf_counter_ns();base=call(ctrl,state,cam,col);bg,br,bl,paths=cache(base,state,cam);alpha=base[2][1].contiguous();base8=m.quantize(base[1]);torch.cuda.synchronize();S=(time.perf_counter_ns()-t0)/1e6
        rows.append(dict(scene=scene,setup_wall_ms=S,retained_cache_and_PNG_bytes=storage_bytes(list(base)+[col,alpha,base8]),GT_bytes_excluded=True,
            separate_process=True,original_setup_not_reconstructed=True))
        del w,state,col,base,bg,br,bl,paths,alpha,base8;gc.collect();torch.cuda.empty_cache()
    write_new(TIM/'setup_addendum.json',dict(rows=rows,native_forwards=2,telemetry_before=before,telemetry_after=telemetry(),training=False,
        scope='New one-shot observed setup for illustrative cost estimates, not original-run setup'))
    print(json.dumps(dict(rows=rows,new_forwards=2)),flush=True)

if __name__=='__main__':main()
