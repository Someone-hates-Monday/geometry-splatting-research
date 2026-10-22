"""Frozen greedy equal-budget policy and full native joint references, no optimizer."""
import argparse,gc,hashlib,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_joint_policy_data_v29 import ROOT,OUT,SOURCE,OLD,sha,write
from tgs_split_color_common_v14 import HELPER,module,ahash,pack,colors,CrossWorkload,exact
from tgs_aux_ablation_common_v21 import control
from tgs_batch_score_common_v22 import bank,child_only,backend as planner_backend
from tgs_packed_query_common_v25 import backend as emitter_backend,dynamic_layout,emit
from tgs_shared_image_loss_common_v26 import backend as loss_backend,evaluate
from tgs_local_image_loss_common_v24 import layout,device_layout
from tgs_joint_edit_common_v27 import expand,overlaps
from run_tgs_local_split_v19 import cache
from run_tgs_real_events import camera_tensors
from trace_tgs_window_scene_v15 import call
POLICIES=['signed_mixed','signed_MSE','perturbation','largest_min_area','hash']
def key(scene,i):return hashlib.sha256(f'tgs_joint_policy_v29/{scene}/{i}'.encode()).hexdigest()
def mix(s):return float(.8*s[1]-.2*s[2])
def rows(scene,cohort):return [r for r in json.loads((OUT/'input_plan.json').read_text())['rows'] if r['scene']==scene and r['cohort']==cohort]
def freeze():
    files=[Path(__file__),ROOT/'tgs_joint_policy_data_v29.py',ROOT/'TGS_JOINT_POLICY_PROTOCOL_V29.md',OUT/'input_plan.json',ROOT/'TGS_JOINT_EDIT_LOCALITY_NOTE_V27.md',ROOT/'TGS_POLICY_REFERENCE_ADDENDUM_V28_R1.md',ROOT/'tgs_joint_edit_common_v27.py',OLD/'execution_plan.json',OLD/'evidence_manifest.json']
    files.extend(ROOT/n for n in json.loads((OLD/'execution_plan.json').read_text())['inputs'])
    for scene in ['flowers','room']:files.append(OLD/scene/'candidate_pool.json')
    write(OUT/'execution_plan.json',dict(inputs={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},pool_size=256,edits=8,batch=8,native_forward_cap=52,finite_precision_contract='fixed_external_FP32_colors_child_only_SH',training=False,backward=False,accepted_mutations=0))
    print('JOINT_POLICY_EXECUTION_FROZEN',flush=True)
def frozen():
    p=json.loads((OUT/'execution_plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p
def baseline(ctrl,helper,state,row):
    H,W={'flowers':(828,1256),'room':(1038,1557)}[row['scene']];cam=camera_tensors(row['camera'],W,H);col=colors(helper,state,cam);native=call(ctrl,state,cam,col);torch.cuda.synchronize();assert torch.isfinite(native[1]).all()
    bg,br,bl,_=cache(native,state,cam);B=native[1].cpu().numpy();dest=OUT/row['scene']/row['view'];dest.mkdir(parents=True);np.save(dest/'baseline_rgb.npy',B,allow_pickle=False)
    gt=np.load(ROOT/row['GT']);assert gt.shape==(H,W,3);gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda();fl=device_layout(layout([[0,0,W,H]],W,H))
    return dict(row=row,cam=cam,col=col,native=native,bg=bg,br=br,bl=bl,base=native[1],B=B,dest=dest,gt8=gt8,fl=fl,shared=None)
def prep_loss(m,q):
    if q['shared'] is None:
        a=m.prepare(q['base'],q['gt8'],q['fl']['weights']);assert not any(a[1].cpu().tolist());q['shared']=a[0]
def pool(scene,w,qs):
    areas=[];vis=[]
    for q in qs:
        g=q['bg'];a=((g[8][:,0].long()-g[7][:,0].long())*(g[8][:,1].long()-g[7][:,1].long())*256).cpu().numpy();v=(g[6]>0).cpu().numpy();a[~v]=0;areas.append(a);vis.append(v)
    minimum=np.minimum(*areas);maximum=np.maximum(*areas);H,W=qs[0]['B'].shape[1:];op=w.opacity.detach().cpu().numpy().reshape(-1);mask=(op>=.01)&vis[0]&vis[1]&(minimum>0)&(maximum<=H*W/4)
    prior=json.loads((ROOT/'experiments/tgs_local_stress_v20/results.json').read_text())['rows'];excluded={r['parent_id'] for r in prior if r['scene']==scene};excluded.update(r['parent_id'] for r in json.loads((OLD/scene/'candidate_pool.json').read_text())['rows']);mask[sorted(excluded)]=False
    ids=np.flatnonzero(mask);ids=ids[np.lexsort((ids,minimum[ids]))];dest=OUT/scene;np.savez_compressed(dest/'eligible_geometry.npz',ids=ids,minimum_area=minimum[ids],maximum_area=maximum[ids]);selected=[]
    for b in range(4):
        group=ids[len(ids)*b//4:len(ids)*(b+1)//4];assert len(group)>=64,(scene,b,len(group))
        for i in sorted(map(int,group),key=lambda i:(key(scene,i),i))[:64]:selected.append(dict(parent_id=i,area_bin=b,minimum_area=int(minimum[i]),maximum_area=int(maximum[i]),hash=key(scene,i),appearance='moved'))
    assert len(selected)==256 and len({r['parent_id'] for r in selected})==256
    write(dest/'candidate_pool.json',dict(rows=selected,eligible_count=len(ids),all_eligible_sha256=sha(dest/'eligible_geometry.npz'),excluded_ids=sorted(excluded),both_calibration_parent_radii_positive=True,loss_scores_seen=False));print(json.dumps(dict(stage='JOINT_POOL_FROZEN',scene=scene,eligible=len(ids))),flush=True);return selected
def emit_indices(planner,emitter,q,indices):
    gidx=torch.tensor([4*i+j for i in indices for j in range(4)],device='cuda',dtype=torch.int64);cg=[g.index_select(0,gidx).contiguous() for g in q['cg']];parents=q['parents'][indices].contiguous();cc=q['cc'].index_select(0,gidx).contiguous();H,W=q['B'].shape[1:]
    boxes,offsets,d,rois=dynamic_layout(planner,q['bg'],cg,parents,W,H);out=emit(emitter,q['bg'],q['br'],q['bl'],q['col'],cg,cc,parents,boxes,offsets,q['base'],d);torch.cuda.synchronize();assert not any(out[1].cpu().tolist());return out[0],d,rois
def score_view(ctrl,helper,planner,emitter,losser,state,q,cases):
    prep_loss(losser,q);bs,cc,parents,small,ccl=bank(state,q['col'],q['cam'],cases,helper);tiny=call(ctrl,bs,q['cam'],cc);torch.cuda.synchronize();cg,_,_,_=cache(tiny,bs,q['cam']);q.update(bank_native=tiny,cg=cg,cc=cc,parents=parents,small=small,child_colors=ccl);table=[];H,W=q['B'].shape[1:]
    for start in range(0,256,8):
        patch,d,rois=emit_indices(planner,emitter,q,list(range(start,start+8)));s=evaluate(losser,q['base'],q['gt8'],patch,d,q['shared'],False);torch.cuda.synchronize();assert not any(s[1].cpu().tolist());scores=s[0].cpu().numpy();packed=patch.cpu().numpy();prefix=d['host']['prefix'][0]
        for j,R in enumerate(rois):
            x0,y0,x1,y1=R;P=packed[int(prefix[j]):int(prefix[j+1])].reshape(3,y1-y0,x1-x0);D=P.astype(float)-q['B'][:,y0:y1,x0:x1].astype(float)
            table.append(dict(parent_id=cases[start+j]['parent_id'],ROI=R,A=expand(R,W,H),score=scores[j].tolist(),mixed=mix(scores[j]),perturbation=float(np.sum(D*D)/(3*H*W)),patch_sha256=ahash(P),RGB_changed=not exact(P,q['B'][:,y0:y1,x0:x1])))
        if (start+8)%64==0:print(json.dumps(dict(stage='JOINT_CALIB_UNARIES',scene=q['row']['scene'],view=q['row']['view'],candidates=start+8)),flush=True)
        del patch,d,s,scores,packed,P,D
    q['table']=table;write(q['dest']/'unary_scores.json',dict(rows=table,baseline_rgb_sha256=ahash(q['B']),GT_sha256=sha(ROOT/q['row']['GT']),new_native_forwards=2));return table
def select(scene,cases,tables,initial):
    aggregate=[]
    for i,p in enumerate(cases):
        s=np.mean([t[i]['score'] for t in tables],axis=0);aggregate.append(dict(p,mean_score=s.tolist(),mean_mixed=mix(s),mean_perturbation=float(np.mean([t[i]['perturbation'] for t in tables]))))
    criteria={'signed_mixed':lambda r:r['mean_mixed'],'signed_MSE':lambda r:r['mean_score'][0],'perturbation':lambda r:r['mean_perturbation'],'largest_min_area':lambda r:-r['minimum_area'],'hash':lambda r:r['hash']};chosen={};trace={}
    pos={r['parent_id']:i for i,r in enumerate(cases)}
    for name in POLICIES:
        accepted=[];skipped=0
        for r in sorted(aggregate,key=lambda r:(criteria[name](r),r['parent_id'])):
            i=pos[r['parent_id']]
            if any(overlaps(tables[v][i]['A'],tables[v][j]['A']) for j in accepted for v in range(2)):skipped+=1;continue
            accepted.append(i)
            if len(accepted)==8:break
        assert len(accepted)==8,(scene,name,len(accepted));chosen[name]=sorted(cases[i]['parent_id'] for i in accepted);trace[name]=dict(greedy_order=[cases[i]['parent_id'] for i in accepted],conflict_skips=skipped,predicted_mean_score=np.sum([aggregate[i]['mean_score'] for i in accepted],axis=0).tolist(),predicted_mean_mixed=float(sum(aggregate[i]['mean_mixed'] for i in accepted)))
    result=dict(scene=scene,rows=aggregate,selected=chosen,trace=trace,guarded_signed_mixed=chosen['signed_mixed'] if trace['signed_mixed']['predicted_mean_mixed'] < -1e-13 else [],model_inputs=initial,calibration_only=True,joint_native_outcomes_seen=False,heldout_payloads_seen=False)
    write(OUT/scene/'choices_scene.json',result);print(json.dumps(dict(stage='JOINT_SCENE_CHOICES_FROZEN',scene=scene,sets=chosen)),flush=True);return result
def selected_patches(planner,emitter,q,cases,sc):
    lookup={r['parent_id']:i for i,r in enumerate(cases)};ids=sorted({p for ps in sc['selected'].values() for p in ps});patches={}
    for start in range(0,len(ids),8):
        group=ids[start:start+8];P,d,rs=emit_indices(planner,emitter,q,[lookup[p] for p in group]);packed=P.cpu().numpy();prefix=d['host']['prefix'][0]
        for j,p in enumerate(group):
            R=rs[j];x0,y0,x1,y1=R;patch=packed[int(prefix[j]):int(prefix[j+1])].reshape(3,y1-y0,x1-x0).copy();assert ahash(patch)==q['table'][lookup[p]]['patch_sha256'];patches[p]=dict(ROI=R,rgb=patch)
            np.savez_compressed(q['dest']/f'unary_{p}.npz',bbox=np.array(R,dtype=np.int32),rgb=patch)
    return patches
def joint_state(state,col,ids,helper,cam):
    assert ids==sorted(set(ids)) and len(ids)==8;keep=torch.ones(len(state['points']),device='cuda',dtype=torch.bool);keep[ids]=False;children=[child_only(state,p) for p in ids];cc=[colors(helper,c,cam) for c in children]
    trial=pack(*[torch.cat([state[k][keep],*[c[k] for c in children]]).contiguous() for k in ['points','sigma','opacity','sh']]);tc=torch.cat([col[keep],*cc]).contiguous();return trial,tc
def unique_sets(sc):return sorted({tuple(ps) for ps in sc['selected'].values()})
def save_delta(dest,B,X):
    changed=np.any(B.view(np.uint32)!=X.view(np.uint32),axis=0);ys,xs=np.nonzero(changed);R=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1] if len(xs) else [0,0,0,0];x0,y0,x1,y1=R
    rebuilt=B.copy();rebuilt[:,y0:y1,x0:x1]=X[:,y0:y1,x0:x1];assert exact(rebuilt,X);np.savez_compressed(dest,bbox=np.array(R,dtype=np.int32),rgb=X[:,y0:y1,x0:x1]);return int(changed.sum())
def joint_refs(ctrl,helper,losser,state,q,sc,patches=None):
    prep_loss(losser,q);outrows=[];lookup={r['parent_id']:i for i,r in enumerate(sc['rows'])}
    for ji,ps in enumerate(unique_sets(sc)):
        ids=list(ps);trial,tc=joint_state(state,q['col'],ids,helper,q['cam']);native=call(ctrl,trial,q['cam'],tc);torch.cuda.synchronize();X=native[1].cpu().numpy();assert np.isfinite(X).all()
        pred=None
        if patches is not None:
            expected=q['B'].copy()
            for p in ids:x0,y0,x1,y1=patches[p]['ROI'];expected[:,y0:y1,x0:x1]=patches[p]['rgb']
            assert exact(expected,X),(q['row']['scene'],q['row']['view'],ji,'joint mosaic');pred=np.sum([q['table'][lookup[p]]['score'] for p in ids],axis=0)
        scored=evaluate(losser,q['base'],q['gt8'],native[1].reshape(-1),q['fl'],q['shared'],False);torch.cuda.synchronize();assert not any(scored[1].cpu().tolist());score=scored[0][0].cpu().numpy()
        if pred is not None:assert np.all(np.abs(pred-score)<=1e-13+1e-9*np.abs(score)),(pred,score)
        target=q['dest']/f'joint_{ji}.npz';changed=save_delta(target,q['B'],X);outrows.append(dict(joint=ji,parents=ids,policies=[k for k,v in sc['selected'].items() if v==ids],score=score.tolist(),mixed=mix(score),full_rgb_sha256=ahash(X),patch_sha256=sha(target),RGB_changed_pixels=changed,calibration_mosaic_literal=True if patches is not None else None,predicted_unary_sum=pred.tolist() if pred is not None else None))
        print(json.dumps(dict(stage='FULL_JOINT_PASSED',scene=q['row']['scene'],view=q['row']['view'],joint=ji)),flush=True);del trial,tc,native,X,scored;gc.collect()
    write(q['dest']/'joint_results.json',dict(rows=outrows,baseline_rgb_sha256=ahash(q['B']),GT_sha256=sha(ROOT/q['row']['GT']),new_native_joint_forwards=len(outrows),training=False));return len(outrows)

@torch.no_grad()
def calibrate():
    frozen();assert not (SOURCE/'manifest.json').exists();ctrl=control();helper=module(HELPER);planner=planner_backend();emitter=emitter_backend();losser=loss_backend();scenes=[];calls=0
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh);qs=[baseline(ctrl,helper,state,r) for r in rows(scene,'calibration')];calls+=2;cases=pool(scene,w,qs);tables=[]
        for q in qs:tables.append(score_view(ctrl,helper,planner,emitter,losser,state,q,cases));calls+=1
        sc=select(scene,cases,tables,initial);scenes.append(sc)
        for q in qs:
            patches=selected_patches(planner,emitter,q,cases,sc);calls+=joint_refs(ctrl,helper,losser,state,q,sc,patches);del patches
        assert initial==w.input_hashes();write(OUT/scene/'calibration_receipt.json',dict(model_inputs_unchanged=True,choices_scene_sha256=sha(OUT/scene/'choices_scene.json'),joint_mosaics_literal=True))
        del qs,tables,w,state;gc.collect();torch.cuda.empty_cache()
    frozen();write(OUT/'choices.json',dict(scenes=scenes,calibration_only=True,heldout_payloads_seen=False,training=False,accepted_mutations=0));write(OUT/'calibration_results.json',dict(all_passed=True,new_native_forwards=calls,choices_sha256=sha(OUT/'choices.json'),training=False));print(json.dumps(dict(stage='ALL_JOINT_CALIBRATION_PASSED',native_calls=calls)),flush=True)

@torch.no_grad()
def heldout():
    frozen();assert json.loads((OUT/'heldout_identity.json').read_text())['all_passed'];choices=json.loads((OUT/'choices.json').read_text());assert json.loads((SOURCE/'manifest.json').read_text())['choices_sha256']==sha(OUT/'choices.json');ctrl=control();helper=module(HELPER);losser=loss_backend();calls=0
    for sc in choices['scenes']:
        scene=sc['scene'];w=CrossWorkload(scene);initial=w.input_hashes();assert initial==sc['model_inputs'];state=pack(w.points,w.sigma,w.opacity,w.sh)
        for row in rows(scene,'heldout'):
            q=baseline(ctrl,helper,state,row);calls+=1;calls+=joint_refs(ctrl,helper,losser,state,q,sc);del q;gc.collect();torch.cuda.empty_cache()
        assert initial==w.input_hashes();del w,state;gc.collect();torch.cuda.empty_cache()
    total=calls+json.loads((OUT/'calibration_results.json').read_text())['new_native_forwards'];assert total<=52;write(OUT/'heldout_results.json',dict(all_passed=True,new_native_forwards=calls,total_native_forwards=total,choices_sha256=sha(OUT/'choices.json'),model_inputs_unchanged=True,training=False));print(json.dumps(dict(stage='ALL_NEW_HELDOUT_JOINT_REFERENCES_PASSED',native_total=total)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','calibrate','heldout']);a=p.parse_args()
    try:globals()[a.action]()
    except BaseException:
        with (OUT/(a.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
