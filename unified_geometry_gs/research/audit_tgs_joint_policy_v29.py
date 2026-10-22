"""CPU replay of frozen selection, joint mosaics and full image-loss references."""
import gc,hashlib,json,math,traceback
from pathlib import Path
import numpy as np
from tgs_joint_policy_data_v29 import ROOT,OUT,SOURCE,sha,write
from tgs_image_metrics_v11 import ssim_map
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def overlap(a,b):return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3] and a[0]<a[2] and b[0]<b[2] and a[1]<a[3] and b[1]<b[3]
def main():
    choices=json.loads((OUT/'choices.json').read_text());assert choices['calibration_only'] and not choices['heldout_payloads_seen'];held=json.loads((OUT/'heldout_results.json').read_text());assert held['all_passed'] and held['total_native_forwards']<=52
    plan=json.loads((OUT/'input_plan.json').read_text());inputs=[Path(__file__),ROOT/'tgs_image_metrics_v11.py',OUT/'input_plan.json',OUT/'execution_plan.json',OUT/'choices.json',OUT/'calibration_results.json',OUT/'heldout_results.json',OUT/'heldout_identity.json',SOURCE/'manifest.json']
    for sc in choices['scenes']:
        folder=OUT/sc['scene'];inputs.extend(folder/n for n in ['eligible_geometry.npz','candidate_pool.json','choices_scene.json','calibration_receipt.json'])
    for r in plan['rows']:
        dest=OUT/r['scene']/r['view'];inputs.extend(dest/n for n in ['baseline_rgb.npy','joint_results.json']);inputs.append(ROOT/r['GT']);inputs.extend(dest.glob('*.npz'))
        if r['cohort']=='calibration':inputs.append(dest/'unary_scores.json')
    write(OUT/'audit_plan.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(inputs))},score_atol=1e-13,score_rtol=1e-9,new_native_forwards=0,posthoc_audit_not_retuning=True))
    selection_checks=[];out=[];baselines=[];max_error=0.;max_add_error=0.
    for sc in choices['scenes']:
        scene=sc['scene'];pool=json.loads((OUT/scene/'candidate_pool.json').read_text())['rows'];assert len(pool)==256;tables=[json.loads((OUT/scene/f'calibration_{v}/unary_scores.json').read_text())['rows'] for v in range(2)];lookup={p['parent_id']:i for i,p in enumerate(pool)}
        eligible=np.load(OUT/scene/'eligible_geometry.npz');ids=eligible['ids'];mina=eligible['minimum_area'];maxa=eligible['maximum_area'];assert np.all(mina>0) and np.all(mina<=maxa)
        for b in range(4):
            group=ids[len(ids)*b//4:len(ids)*(b+1)//4];selected=sorted(map(int,group),key=lambda i:(hashlib.sha256(f'tgs_joint_policy_v29/{scene}/{i}'.encode()).hexdigest(),i))[:64]
            assert selected==[p['parent_id'] for p in pool if p['area_bin']==b]
        aggregate=[]
        for i,p in enumerate(pool):
            assert all(t[i]['parent_id']==p['parent_id'] for t in tables);s=np.mean([t[i]['score'] for t in tables],axis=0);aggregate.append(dict(p,mean_score=s,mean_mixed=float(.8*s[1]-.2*s[2]),mean_perturbation=float(np.mean([t[i]['perturbation'] for t in tables]))))
        criteria={'signed_mixed':lambda r:r['mean_mixed'],'signed_MSE':lambda r:r['mean_score'][0],'perturbation':lambda r:r['mean_perturbation'],'largest_min_area':lambda r:-r['minimum_area'],'hash':lambda r:r['hash']}
        for name in criteria:
            selected=[]
            for p in sorted(aggregate,key=lambda r:(criteria[name](r),r['parent_id'])):
                i=lookup[p['parent_id']]
                if any(overlap(tables[v][i]['A'],tables[v][j]['A']) for j in selected for v in range(2)):continue
                selected.append(i)
                if len(selected)==8:break
            assert sorted(pool[i]['parent_id'] for i in selected)==sc['selected'][name] and len(selected)==8
        selection_checks.append(dict(scene=scene,all_five_greedy_selections_reproduced=True,all_budgets_eight=True,all_calibration_SSIM_supports_disjoint=True,pool_hash_sampling_reproduced=True))
        for r in [r for r in plan['rows'] if r['scene']==scene]:
            view=r['view'];dest=OUT/scene/view;B=np.load(dest/'baseline_rgb.npy');tab=json.loads((dest/'joint_results.json').read_text());assert ahash(B)==tab['baseline_rgb_sha256'];GT=np.load(ROOT/r['GT']);G=GT.astype(float)/255.;b=B.transpose(1,2,0).astype(float);be=(b-G)**2;ba=np.abs(b-G);bs=ssim_map(b,G);baseMSE=float(be.mean());baselines.append(dict(scene=scene,view=view,MSE=baseMSE,L1=float(ba.mean()),SSIM=float(bs.mean()),raw_PSNR_dB=-10*math.log10(baseMSE)))
            for ref in tab['rows']:
                path=dest/f"joint_{ref['joint']}.npz";assert sha(path)==ref['patch_sha256'];d=np.load(path);x0,y0,x1,y1=d['bbox'].tolist();X=B.copy();X[:,y0:y1,x0:x1]=d['rgb'];assert ahash(X)==ref['full_rgb_sha256']
                if r['cohort']=='calibration':
                    expected=B.copy()
                    for p in ref['parents']:
                        d1=np.load(dest/f'unary_{p}.npz');R=d1['bbox'].tolist();rx0,ry0,rx1,ry1=R;assert R==tables[int(view[-1])][lookup[p]]['ROI'];assert ahash(d1['rgb'])==tables[int(view[-1])][lookup[p]]['patch_sha256'];expected[:,ry0:ry1,rx0:rx1]=d1['rgb']
                    assert ahash(expected)==ref['full_rgb_sha256'];del expected
                x=X.transpose(1,2,0).astype(float);s=np.array([np.mean((x-G)**2-be),np.mean(np.abs(x-G)-ba),np.mean(ssim_map(x,G)-bs)]);cached=np.array(ref['score']);err=np.abs(s-cached);assert np.all(err<=1e-13+1e-9*np.abs(s));max_error=max(max_error,float(err.max()))
                if ref['predicted_unary_sum'] is not None:
                    ae=np.abs(s-np.array(ref['predicted_unary_sum']));assert np.all(ae<=1e-13+1e-9*np.abs(s));max_add_error=max(max_add_error,float(ae.max()))
                out.append(dict(scene=scene,view=view,joint=ref['joint'],parents=ref['parents'],policies=ref['policies'],CPU_score=s.tolist(),CPU_mixed=float(.8*s[1]-.2*s[2]),cached_GPU_score=cached.tolist(),max_error=float(err.max()),raw_PSNR_change_dB=-10*math.log10((baseMSE+s[0])/baseMSE),RGB_changed_pixels=ref['RGB_changed_pixels'],full_RGB_hash_verified=True,calibration_mosaic_literal=True if r['cohort']=='calibration' else None))
                del X,x;gc.collect()
            print(json.dumps(dict(stage='JOINT_CPU_VIEW_AUDITED',scene=scene,view=view,joints=len(tab['rows']))),flush=True);del B,GT,G,b,be,ba,bs;gc.collect()
    write(OUT/'independent_audit.json',dict(all_passed=True,rows=out,baselines=baselines,selection_checks=selection_checks,maximum_CPU_GPU_score_error=max_error,maximum_calibration_unary_sum_CPU_error=max_add_error,total_native_forwards=held['total_native_forwards'],audited_joint_references=len(out),new_native_forwards=0,training=False,accepted_mutations=0))
    print(json.dumps(dict(stage='ALL_JOINT_CPU_AUDITS_PASSED',references=len(out),max_error=max_error)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        with (OUT/'independent_audit_failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
