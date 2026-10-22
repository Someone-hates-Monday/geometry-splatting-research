"""Two full-scene causal forward comparisons of the upper-bound-only change.

No GT, optimizer, training or performance timing. Measures image/coverage changes,
not image-quality gains, using the same frozen published model and train cameras.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from benchmark_tgs_backward import ROOT,Workload
from tgs_mixed_reference import load_module

OUT=ROOT/'experiments/tgs_radius_contraction_v1/tile_coverage'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target=OUT/'real_upper_only.json';assert not target.exists()
    build=json.loads((OUT/'upper_fix_build/build_receipt.json').read_text());path=Path(build['module_file'])
    assert sha(path)==build['module_sha256']
    spec=importlib.util.spec_from_file_location(path.stem,path);fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixed)
    control=load_module('fixed');work=Workload()
    rows=[]
    plan=dict(workload=work.metadata,cameras=work.camera_metadata,changed='four exclusive tile upper bounds only',
              reference_module_sha256=sha(Path(control.__file__)),fixed_module_sha256=build['module_sha256'],
              purpose='causal forward/coverage change; no quality or timing claim',training=False,
              script_sha256=sha(Path(__file__)))
    (OUT/'real_upper_only_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for ci in range(2):
        a=work.forward(control,ci);b=work.forward(fixed,ci);torch.cuda.synchronize()
        before=a[1].cpu().numpy();after=b[1].cpu().numpy()
        aa=a[2][1].cpu().numpy();ab=b[2][1].cpu().numpy()
        ra=a[3].cpu().numpy();rb=b[3].cpu().numpy()
        assert np.isfinite(before).all() and np.isfinite(after).all()
        d=np.abs(after.astype(float)-before.astype(float))
        ad=ab.astype(float)-aa.astype(float)
        historical=ROOT/f'experiments/tgs_real_event_v1/baseline_{ci+2}.npz'
        with np.load(historical) as z:old=z['fixed_rgb'];cached_difference=float(np.max(np.abs(before-old)))
        row=dict(camera=ci,image=work.camera_metadata[ci]['img_name'],
            original_tile_instances=int(a[0]),fixed_tile_instances=int(b[0]),
            original_visible=int((ra>0).sum()),fixed_visible=int((rb>0).sum()),
            newly_visible=int(((ra==0)&(rb>0)).sum()),lost_visibility=int(((ra>0)&(rb==0)).sum()),
            rgb_max_abs=float(d.max()),rgb_mean_abs=float(d.mean()),
            pixels_changed_any=int((d.max(0)>0).sum()),pixels_changed_above_3e_5=int((d.max(0)>3e-5).sum()),
            alpha_positive_pixels=int((ad>3e-5).sum()),alpha_negative_pixels=int((ad<-3e-5).sum()),
            alpha_max_change=float(ad.max()),alpha_min_change=float(ad.min()),
            historical_fixed_rgb_max_difference=cached_difference,historical_baseline_sha256=sha(historical))
        rows.append(row)
        np.savez_compressed(OUT/f'real_camera_{ci}_upper_only.npz',before_rgb=before,after_rgb=after,
                            before_alpha=aa,after_alpha=ab,before_radii=ra,after_radii=rb)
        print(json.dumps(row),flush=True)
        del a,b
    target.write_text(json.dumps(dict(plan_sha256=sha(OUT/'real_upper_only_plan.json'),rows=rows,
        quality_gain_established=False,training=False,performance_measurement=False),indent=2)+'\n')


if __name__=='__main__':main()
