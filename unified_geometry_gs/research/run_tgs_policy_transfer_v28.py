"""Frozen one-edit selection; test payloads and scores require a choice receipt."""
import argparse,gc,hashlib,json,traceback
from pathlib import Path
import numpy as np
import torch
from prepare_tgs_policy_data_v28 import ROOT,OUT,SOURCE,sha,write
from tgs_split_color_common_v14 import HELPER,ahash,module,pack,split,colors,CrossWorkload,exact
from tgs_aux_ablation_common_v21 import control
from tgs_batch_score_common_v22 import backend as planner_backend,bank
from tgs_packed_query_common_v25 import backend as emitter_backend,dynamic_layout,emit
from tgs_shared_image_loss_common_v26 import backend as loss_backend,evaluate
from tgs_local_image_loss_common_v24 import layout,device_layout
from run_tgs_local_split_v19 import cache
from run_tgs_real_events import camera_tensors
from trace_tgs_window_scene_v15 import call

def key(scene,i):return hashlib.sha256(f'tgs_policy_transfer_v28/{scene}/{i}'.encode()).hexdigest()
def mixed(s):return .8*s[1]-.2*s[2]
def freeze():
    assert json.loads((OUT/'calibration_identity.json').read_text())['all_passed']
    files=[Path(__file__),ROOT/'prepare_tgs_policy_data_v28.py',ROOT/'TGS_POLICY_TRANSFER_PROTOCOL_V28.md',OUT/'input_plan.json',OUT/'calibration_identity.json',SOURCE/'calibration_manifest.json']
    for f in ['experiments/tgs_packed_query_v25/plan.json','experiments/tgs_packed_query_v25/runtime_dependency_addendum.json','experiments/tgs_shared_loss_v26/plan.json']:
        p=ROOT/f;files.append(p);files.extend(ROOT/n for n in json.loads(p.read_text())['inputs'])
    for f in ['tgs_aux_ablation_common_v21.py','tgs_batch_score_common_v22.py','tgs_packed_query_common_v25.py','tgs_shared_image_loss_common_v26.py','tgs_local_image_loss_common_v24.py','trace_tgs_window_scene_v15.py','tgs_split_color_common_v14.py']:
        files.append(ROOT/f)
    for row in json.loads((OUT/'input_plan.json').read_text())['rows']:
        if row['cohort']=='calibration':files.append(OUT/row['scene']/row['view']/'gt_rgb.npy')
    write(OUT/'execution_plan.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},native_forward_cap=56,candidates_per_scene=32,batch_size=8,training=False,backward=False,accepted_mutations=0,heldout_payloads_seen=False))
    print('POLICY_EXECUTION_PLAN_FROZEN',flush=True)
def frozen():
    p=json.loads((OUT/'execution_plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p
def view_rows(scene,cohort):return [r for r in json.loads((OUT/'input_plan.json').read_text())['rows'] if r['scene']==scene and r['cohort']==cohort]
def camera(row):
    H,W={'flowers':(828,1256),'room':(1038,1557)}[row['scene']]
    return camera_tensors(row['camera'],W,H)
def candidate_pool(scene,w,cached):
    area=np.zeros(w.n,dtype=np.int64)
    for q in cached:
        g=q['bg'];a=((g[8][:,0]-g[7][:,0]).long()*(g[8][:,1]-g[7][:,1]).long()*256).cpu().numpy()
        # Native rectangles count16x16 tiles, including clipped boundary tiles.
        a[g[6].cpu().numpy()<=0]=0;area=np.maximum(area,a)
    H,W=cached[0]['base'].shape[1:];op=w.opacity.detach().cpu().numpy().reshape(-1)
    mask=(op>=.01)&(area>0)&(area<=H*W/4)
    old=json.loads((ROOT/'experiments/tgs_local_stress_v20/results.json').read_text())['rows']
    excluded=sorted({r['parent_id'] for r in old if r['scene']==scene});mask[excluded]=False
    ids=np.flatnonzero(mask);order=np.lexsort((ids,area[ids]));ids=ids[order];assert len(ids)>=32
    dest=OUT/scene;np.savez_compressed(dest/'eligible_geometry.npz',ids=ids,maximum_native_rectangle_area=area[ids])
    rows=[]
    for b in range(4):
        group=ids[len(ids)*b//4:len(ids)*(b+1)//4];assert len(group)>=8
        selected=sorted(map(int,group),key=lambda i:(key(scene,i),i))[:8]
        for i in selected:rows.append(dict(parent_id=i,area_bin=b,maximum_calibration_native_rectangle_area=int(area[i]),hash=key(scene,i),appearance='moved'))
    assert len(rows)==32 and len({r['parent_id'] for r in rows})==32
    write(dest/'candidate_pool.json',dict(rows=rows,eligible_count=len(ids),excluded_prior_stress_ids=excluded,eligible_sha256=sha(dest/'eligible_geometry.npz'),calibration_geometry_only=True,loss_scores_seen=False))
    print(json.dumps(dict(stage='POOL_FROZEN',scene=scene,eligible=len(ids),parents=[r['parent_id'] for r in rows])),flush=True)
    return rows
def baseline(ctrl,helper,state,row):
    dest=OUT/row['scene']/row['view'];cam=camera(row);col=colors(helper,state,cam);native=call(ctrl,state,cam,col);torch.cuda.synchronize()
    bg,br,bl,_=cache(native,state,cam);base=native[1];B=base.cpu().numpy();np.save(dest/'baseline_rgb.npy',B,allow_pickle=False)
    return dict(row=row,cam=cam,col=col,native=native,bg=bg,br=br,bl=bl,base=base,B=B)
def score_view(ctrl,helper,planner,emitter,losser,state,q,cases):
    row=q['row'];dest=OUT/row['scene']/row['view'];H,W=q['base'].shape[1:]
    gt=np.load(dest/'gt_rgb.npy');gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda()
    d0=device_layout(layout([[0,0,W,H]],W,H));prep=losser.prepare(q['base'],gt8,d0['weights']);assert not any(prep[1].cpu().tolist());shared=prep[0]
    bs,cc,parents,_,_=bank(state,q['col'],q['cam'],cases,helper);tiny=call(ctrl,bs,q['cam'],cc);torch.cuda.synchronize();cg,_,_,_=cache(tiny,bs,q['cam']);rows=[]
    for start in range(0,len(cases),8):
        end=min(start+8,len(cases));subcg=[x[4*start:4*end].contiguous() for x in cg];subparents=parents[start:end].contiguous();subcc=cc[4*start:4*end].contiguous()
        boxes,offsets,d,rois=dynamic_layout(planner,q['bg'],subcg,subparents,W,H)
        emitted=emit(emitter,q['bg'],q['br'],q['bl'],q['col'],subcg,subcc,subparents,boxes,offsets,q['base'],d)
        scored=evaluate(losser,q['base'],gt8,emitted[0],d,shared,False);torch.cuda.synchronize()
        assert not any(emitted[1].cpu().tolist()) and not any(scored[1].cpu().tolist())
        scores=scored[0].cpu().numpy();packed=emitted[0].cpu().numpy();prefix=d['host']['prefix'][0]
        for j,r in enumerate(rois):
            x0,y0,x1,y1=r;patch=packed[int(prefix[j]):int(prefix[j+1])].reshape(3,y1-y0,x1-x0)
            full=q['B'].copy();full[:,y0:y1,x0:x1]=patch
            delta=patch.astype(np.float64)-q['B'][:,y0:y1,x0:x1].astype(np.float64)
            perturb=float(np.sum(delta*delta,dtype=np.float64)/(3*H*W))
            assert np.isfinite(scores[j]).all() and np.isfinite(perturb)
            rows.append(dict(parent_id=cases[start+j]['parent_id'],ROI=r,score=scores[j].tolist(),mixed_L1_DSSIM=mixed(scores[j]),perturbation_MSE=perturb,rgb_sha256=ahash(full)))
        print(json.dumps(dict(stage='SCORED_BATCH',scene=row['scene'],view=row['view'],candidates=end)),flush=True)
        del subcg,subparents,subcc,boxes,offsets,d,emitted,scored,packed,full,patch,delta
    write(dest/'scores.json',dict(rows=rows,baseline_RGB_sha256=ahash(q['B']),baseline_file_sha256=sha(dest/'baseline_rgb.npy'),candidate_pool_sha256=sha(OUT/row['scene']/'candidate_pool.json'),GT_sha256=sha(dest/'gt_rgb.npy'),cohort=row['cohort'],new_native_forwards=2,shared_cache_preps=1,full_native_per_candidate=False,training=False))
    return rows

@torch.no_grad()
def calibrate():
    frozen();assert not (SOURCE/'heldout_manifest.json').exists() and not (OUT/'choices.json').exists()
    ctrl=control();helper=module(HELPER);planner=planner_backend();emitter=emitter_backend();losser=loss_backend();selections=[]
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh)
        cached=[baseline(ctrl,helper,state,r) for r in view_rows(scene,'calibration')];pool=candidate_pool(scene,w,cached);tables=[]
        for q in cached:tables.append(score_view(ctrl,helper,planner,emitter,losser,state,q,pool))
        aggregate=[]
        for j,p in enumerate(pool):
            s=np.mean([t[j]['score'] for t in tables],axis=0)
            aggregate.append(dict(p,mean_score=s.tolist(),mean_mixed=mixed(s),mean_perturbation=float(np.mean([t[j]['perturbation_MSE'] for t in tables]))))
        criteria={'signed_mixed':lambda r:r['mean_mixed'],'signed_MSE':lambda r:r['mean_score'][0],'perturbation':lambda r:r['mean_perturbation'],'largest_area':lambda r:-r['maximum_calibration_native_rectangle_area'],'hash':lambda r:r['hash']}
        selected={name:min(aggregate,key=lambda r:(criterion(r),r['parent_id']))['parent_id'] for name,criterion in criteria.items()}
        primary=next(r for r in aggregate if r['parent_id']==selected['signed_mixed'])
        selections.append(dict(scene=scene,rows=aggregate,selected=selected,guarded_signed_mixed=selected['signed_mixed'] if primary['mean_mixed'] < -1e-13 else None,model_inputs=initial,model_inputs_unchanged=initial==w.input_hashes(),candidate_pool_sha256=sha(OUT/scene/'candidate_pool.json'),calibration_scores={r['view']:sha(OUT/scene/r['view']/'scores.json') for r in view_rows(scene,'calibration')}))
        assert initial==w.input_hashes();del cached,tables,w,state;gc.collect();torch.cuda.empty_cache()
    frozen();write(OUT/'choices.json',dict(scenes=selections,calibration_only=True,test_scores_seen=False,heldout_payloads_seen=False,execution_plan_sha256=sha(OUT/'execution_plan.json'),new_native_forwards=8,training=False,accepted_mutations=0))
    print(json.dumps(dict(stage='CHOICES_FROZEN',choices=[dict(scene=r['scene'],selected=r['selected'],guard=r['guarded_signed_mixed']) for r in selections])),flush=True)

@torch.no_grad()
def heldout():
    frozen();choices=json.loads((OUT/'choices.json').read_text());identity=json.loads((OUT/'heldout_identity.json').read_text());assert identity['all_passed']
    assert json.loads((SOURCE/'heldout_manifest.json').read_text())['choices_sha256']==sha(OUT/'choices.json')
    ctrl=control();helper=module(HELPER);planner=planner_backend();emitter=emitter_backend();losser=loss_backend()
    for scene in ['flowers','room']:
        sc=next(s for s in choices['scenes'] if s['scene']==scene);w=CrossWorkload(scene);initial=w.input_hashes();assert initial==sc['model_inputs'];state=pack(w.points,w.sigma,w.opacity,w.sh)
        pool=json.loads((OUT/scene/'candidate_pool.json').read_text())['rows']
        for row in view_rows(scene,'heldout'):
            q=baseline(ctrl,helper,state,row);score_view(ctrl,helper,planner,emitter,losser,state,q,pool);del q;gc.collect();torch.cuda.empty_cache()
        assert initial==w.input_hashes();del w,state;gc.collect();torch.cuda.empty_cache()
    write(OUT/'heldout_receipt.json',dict(all_passed=True,choices_sha256=sha(OUT/'choices.json'),identity_sha256=sha(OUT/'heldout_identity.json'),new_native_forwards=8,model_inputs_unchanged=True,training=False))

@torch.no_grad()
def reference(cohort):
    frozen();choices=json.loads((OUT/'choices.json').read_text());ctrl=control();helper=module(HELPER);rows=[];calls=0
    if cohort=='heldout':assert json.loads((OUT/'heldout_receipt.json').read_text())['all_passed']
    for scene in ['flowers','room']:
        sc=next(s for s in choices['scenes'] if s['scene']==scene);w=CrossWorkload(scene);initial=w.input_hashes();assert initial==sc['model_inputs'];state=pack(w.points,w.sigma,w.opacity,w.sh);ids=sorted(set(sc['selected'].values()))
        for row in view_rows(scene,cohort):
            dest=OUT/scene/row['view'];B=np.load(dest/'baseline_rgb.npy');table=json.loads((dest/'scores.json').read_text());assert ahash(B)==table['baseline_RGB_sha256'];cam=camera(row);col=colors(helper,state,cam)
            for parent in ids:
                ref=next(t for t in table['rows'] if t['parent_id']==parent);trial=split(state,parent);tc=colors(helper,trial,cam);out=call(ctrl,trial,cam,tc);torch.cuda.synchronize();calls+=1;rgb=out[1].cpu().numpy();assert ahash(rgb)==ref['rgb_sha256']
                x0,y0,x1,y1=ref['ROI'];rebuilt=B.copy();rebuilt[:,y0:y1,x0:x1]=rgb[:,y0:y1,x0:x1];assert exact(rebuilt,rgb)
                target=dest/f'parent_{parent}_full_reference.npz';assert not target.exists();np.savez_compressed(target,bbox=np.array(ref['ROI'],dtype=np.int32),rgb=rgb[:,y0:y1,x0:x1])
                rows.append(dict(scene=scene,view=row['view'],parent_id=parent,full_RGB_literal_cached=True,full_rgb_sha256=ahash(rgb),patch_sha256=sha(target),ROI_outside_unchanged=True))
                print(json.dumps(dict(stage='FULL_REFERENCE_PASSED',scene=scene,view=row['view'],parent=parent,native_calls=calls)),flush=True)
                del trial,tc,out,rgb,rebuilt;gc.collect()
        assert initial==w.input_hashes();del w,state;gc.collect();torch.cuda.empty_cache()
    write(OUT/(cohort+'_reference_receipt.json'),dict(all_passed=True,rows=rows,choices_sha256=sha(OUT/'choices.json'),new_native_forwards=calls,training=False,model_inputs_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','calibrate','heldout','reference_calibration','reference_heldout']);a=p.parse_args()
    try:
        if a.action.startswith('reference_'):reference(a.action.split('_',1)[1])
        else:globals()[a.action]()
    except BaseException:
        with (OUT/(a.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
