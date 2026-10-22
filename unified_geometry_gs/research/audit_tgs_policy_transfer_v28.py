"""Independent full-image CPU audit of every selected native reference."""
import gc,json,math,traceback
from pathlib import Path
import numpy as np
from prepare_tgs_policy_data_v28 import ROOT,OUT,sha,write
from tgs_image_metrics_v11 import ssim_map

def ahash(a):return __import__('hashlib').sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    assert not (OUT/'independent_audit.json').exists();choices=json.loads((OUT/'choices.json').read_text());inputs=[Path(__file__),ROOT/'tgs_image_metrics_v11.py',OUT/'choices.json',OUT/'execution_plan.json',OUT/'input_plan.json']
    native=0
    for cohort in ['calibration','heldout']:
        p=OUT/(cohort+'_reference_receipt.json');receipt=json.loads(p.read_text());assert receipt['all_passed'] and receipt['choices_sha256']==sha(OUT/'choices.json');native+=receipt['new_native_forwards'];inputs.append(p)
    native+=choices['new_native_forwards']+json.loads((OUT/'heldout_receipt.json').read_text())['new_native_forwards'];assert native<=56
    for scene in ['flowers','room']:
        for cohort in ['calibration','heldout']:
            for vi in range(2):
                dest=OUT/scene/f'{cohort}_{vi}'
                inputs.extend(dest/n for n in ['scores.json','baseline_rgb.npy','gt_rgb.npy']);inputs.extend(dest.glob('*_full_reference.npz'))
    write(OUT/'audit_plan.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},score_atol=1e-13,score_rtol=1e-9,new_native_forwards=0,post_selection_audit_not_policy_tuning=True))
    rows=[];baselines=[];max_error=0.
    for scene in ['flowers','room']:
        sc=next(s for s in choices['scenes'] if s['scene']==scene)
        for cohort in ['calibration','heldout']:
            for vi in range(2):
                view=f'{cohort}_{vi}';dest=OUT/scene/view;table=json.loads((dest/'scores.json').read_text());B=np.load(dest/'baseline_rgb.npy');gt=np.load(dest/'gt_rgb.npy');assert ahash(B)==table['baseline_RGB_sha256'];G=gt.astype(np.float64)/255.;b=B.transpose(1,2,0).astype(np.float64)
                be=(b-G)**2;ba=np.abs(b-G);bs=ssim_map(b,G);base=dict(scene=scene,view=view,MSE=float(be.mean()),L1=float(ba.mean()),SSIM=float(bs.mean()));baselines.append(base)
                for parent in sorted(set(sc['selected'].values())):
                    ref=next(t for t in table['rows'] if t['parent_id']==parent);data=np.load(dest/f'parent_{parent}_full_reference.npz');box=data['bbox'].tolist();assert box==ref['ROI'];x0,y0,x1,y1=box
                    rgb=B.copy();rgb[:,y0:y1,x0:x1]=data['rgb'];assert ahash(rgb)==ref['rgb_sha256'];x=rgb.transpose(1,2,0).astype(np.float64)
                    s=np.array([np.mean((x-G)**2-be),np.mean(np.abs(x-G)-ba),np.mean(ssim_map(x,G)-bs)],dtype=np.float64);cached=np.array(ref['score']);err=np.abs(s-cached);assert np.all(err<=1e-13+1e-9*np.abs(s)),(scene,view,parent,s,cached)
                    p=float(np.mean((x-b)**2));assert abs(p-ref['perturbation_MSE'])<=1e-13+1e-9*abs(p)
                    max_error=max(max_error,float(err.max()));m=.8*s[1]-.2*s[2]
                    rows.append(dict(scene=scene,view=view,parent_id=parent,CPU_score=s.tolist(),cached_GPU_score=cached.tolist(),score_error=err.tolist(),CPU_mixed=float(m),cached_mixed=ref['mixed_L1_DSSIM'],CPU_perturbation=p,raw_PSNR_change_dB=float(-10*math.log10((base['MSE']+s[0])/base['MSE'])),RGB_hash_verified=True))
                    del rgb,x;gc.collect()
                print(json.dumps(dict(stage='CPU_SELECTED_AUDIT_PASSED',scene=scene,view=view)),flush=True)
                del B,gt,G,b,be,ba,bs;gc.collect()
    write(OUT/'independent_audit.json',dict(all_passed=True,rows=rows,baselines=baselines,maximum_selected_CPU_GPU_error=max_error,total_new_native_forwards=native,new_native_forwards=0,all_selected_references_verified=True,unselected_scores_not_full_native_audited=True,audit_plan_sha256=sha(OUT/'audit_plan.json'),training=False))
    print(json.dumps(dict(stage='ALL_SELECTED_CPU_AUDITS_PASSED',references=len(rows),max_error=max_error,native_calls=native)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'independent_audit_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
