"""All32 real fixed images: finite halo theorem, literal maps, bounded FP64 sums."""
import json,traceback
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,exact
from tgs_image_metrics_v11 import png8,ssim_map
from tgs_batch_score_common_v22 import cpu_sse
OUT=ROOT/'experiments/tgs_view_transfer_v23';V20=ROOT/'experiments/tgs_local_stress_v20'

def dilate(box,r,W,H):
    x0,y0,x1,y1=box
    return [max(0,x0-r),max(0,y0-r),min(W,x1+r),min(H,y1+r)]

def crop(a,box):
    x0,y0,x1,y1=box;return a[y0:y1,x0:x1]

def reduce_check(terms,n,expected):
    assert terms.dtype==np.float64 and np.isfinite(terms).all()
    value=float(terms.sum()/n);m=terms.size;eps=np.finfo(np.float64).eps
    gamma=lambda k:k*eps/(1-k*eps)
    bound=float((gamma(n)+gamma(m)+4*eps)*np.abs(terms).sum()/n)
    residual=abs(value-expected);assert residual<=bound,(residual,bound,value,expected)
    return dict(value=value,full_value=expected,absolute_residual=residual,reduction_roundoff_bound=bound)

def main():
    analysis=json.loads((OUT/'analysis.json').read_text());assert analysis['all_verified'] and len(analysis['rows'])==32
    plan=json.loads((OUT/'plan.json').read_text())
    for n,h in plan['inputs'].items():assert sha(ROOT/n)==h,n
    files=[Path(__file__),ROOT/'TGS_IMAGE_LOSS_LOCALITY_NOTE_V23.md',OUT/'analysis.json',OUT/'results.json',OUT/'plan.json']
    write_new(OUT/'loss_locality_plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},data_informed_addendum=True,
        all32_cases=True,new_GPU_forwards=0,training=False,output_halo=5,input_halo=10))
    rows=[]
    for scene in ['flowers','room']:
        for view in ['fresh','old_0']:
            base=np.load(V20/scene/'baseline.npz')['rgb'] if view=='fresh' else np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy')
            gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy' if view=='fresh' else ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy')
            b=base.transpose(1,2,0).astype(np.float64);g=gt.astype(np.float64)/255.;H,W=g.shape[:2];n=g.size;bs=ssim_map(b,g)
            for en in [x for x in analysis['rows'] if x['scene']==scene and x['view']==view]:
                e=en['event'];pf=V20/scene/f"parent_{en['parent_index']}"/en['appearance']/'query_patch.npz' if view=='fresh' else OUT/scene/f'proposal_{e}_rgb_patch.npz'
                patch=np.load(pf);R=patch['bbox'].tolist();x0,y0,x1,y1=R;trial=base.copy();trial[:,y0:y1,x0:x1]=patch['rgb'];assert ahash(trial)==en['rgb_sha256']
                outside=np.ones((H,W),dtype=bool);outside[y0:y1,x0:x1]=False;assert exact(base[:,outside],trial[:,outside])
                t=trial.transpose(1,2,0).astype(np.float64);ts=ssim_map(t,g);A=dilate(R,5,W,H);J=dilate(A,5,W,H)
                ax0,ay0,ax1,ay1=A;jx0,jy0,_,_=J;select=[ax0-jx0,ay0-jy0,ax1-jx0,ay1-jy0]
                local_t=crop(ssim_map(crop(t,J),crop(g,J)),select);local_b=crop(ssim_map(crop(b,J),crop(g,J)),select)
                assert exact(local_t,crop(ts,A)) and exact(local_b,crop(bs,A))
                mask=np.ones((H,W),dtype=bool);mask[ay0:ay1,ax0:ax1]=False;assert exact(ts[mask],bs[mask])
                er=crop(t,R)-crop(g,R);eb=crop(b,R)-crop(g,R)
                changes=en['changes'];reduced=dict(raw_MSE=reduce_check(er*er-eb*eb,n,changes['raw_MSE']),raw_L1=reduce_check(np.abs(er)-np.abs(eb),n,changes['raw_L1']),raw_SSIM=reduce_check(local_t-local_b,n,changes['raw_SSIM']))
                roi8=png8(trial)[:, :];base8=png8(base)
                roi_delta=cpu_sse(roi8[y0:y1,x0:x1],gt[y0:y1,x0:x1])-cpu_sse(base8[y0:y1,x0:x1],gt[y0:y1,x0:x1]);assert roi_delta==en['delta_SSE']
                rows.append(dict(scene=scene,view=view,event=e,ROI=R,output_region=A,input_region=J,local_maps_literal=True,zero_difference_outside_output_region=True,reductions=reduced,PNG8_SSE_exact=True))
                print(json.dumps(dict(scene=scene,view=view,event=e,halo_literal=True,max_residual=max(v['absolute_residual'] for v in reduced.values()))),flush=True)
    assert len(rows)==32
    write_new(OUT/'loss_locality_results.json',dict(all_passed=True,rows=rows,new_GPU_forwards=0,training=False,
        not_a_speed_or_quality_benchmark=True,source_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'loss_locality_plan.json')))

if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'loss_locality_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
