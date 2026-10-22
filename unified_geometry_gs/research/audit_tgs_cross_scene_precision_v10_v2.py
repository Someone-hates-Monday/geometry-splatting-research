"""Independent full-array VJP metrics and original-oracle recapture on four new inputs."""
import gc,hashlib,json,math,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,OUT,SOURCE,sha,modules,CrossWorkload,decode,state
from audit_tgs_compensated_baseline_diagnostic import anchor_view,rects
from audit_tgs_window_candidates_cache import coarse_cache_audit

def data(folder,key):return np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)

def independent_group(a,b):
    a=a.reshape(-1);b=b.reshape(-1);sx=[];sy=[];sd=[];maximum=0.;finite=True
    for lo in range(0,len(a),131071):
        x=a[lo:lo+131071].astype(np.float64);y=b[lo:lo+131071].astype(np.float64);finite &= bool(np.isfinite(x).all() and np.isfinite(y).all());d=x-y;maximum=max(maximum,float(np.max(np.abs(d),initial=0)));sx.append(float(np.einsum('i,i->',x,x)));sy.append(float(np.einsum('i,i->',y,y)));sd.append(float(np.einsum('i,i->',d,d)))
    if not finite:return dict(finite=False,gate=False,relative=None,absolute_max=None,reference_l2=None)
    ref=math.sqrt(math.fsum(sy));err=math.sqrt(math.fsum(sd));rel=err/max(ref,1e-30);return dict(finite=True,reference_l2=ref,relative=rel,absolute_max=maximum,gate=bool(rel<=1e-5 if ref>1e-10 else maximum<=1e-6))

def main():
    repair=json.loads((OUT/'audit_repair_plan.json').read_text())
    assert repair['script_sha256']==sha(Path(__file__))
    assert repair['original_auditor_sha256']==sha(ROOT/'audit_tgs_cross_scene_precision_v10.py')
    assert repair['diagnostic_sha256']==sha(OUT/'recapture_diagnostic.json')
    assert json.loads((OUT/'recapture_diagnostic.json').read_text())['only_invisible_projection_scratch_differs']
    for filename in ['literal_bit_audit.json','v9_literal_bit_audit.json','v9_FD_sample_bit_audit.json']:
        assert json.loads((OUT/filename).read_text())['literal_bits_equal']
    target=OUT/'precision_audit.json';assert not target.exists();plan=json.loads((OUT/'real_plan.json').read_text());real=json.loads((OUT/'real_results.json').read_text());inputs=json.loads((OUT/'input_audit.json').read_text());assert len(real['rows'])==4 and real['model_inputs_unchanged'] and plan['input_audit_sha256']==sha(OUT/'input_audit.json')
    for name,h in plan['scripts'].items():assert sha(ROOT/name)==h
    assert plan['protocol_sha256']==sha(ROOT/'TGS_CROSS_SCENE_PROTOCOL_V10.md') and real['plan_sha256']==sha(OUT/'real_plan.json')
    for en in json.loads((SOURCE/'manifest.json').read_text())['files']:assert sha(ROOT/en['local'])==en['sha256']
    mods=modules();rows=[]
    for scene in [s['scene'] for s in inputs['scenes']]:
        w=CrossWorkload(scene);assert w.input_hashes()==plan['model_inputs'][scene]
        for ci,cam in enumerate(w.camera_metadata):
            saved=next(r for r in real['rows'] if r['scene']==scene and r['camera']==ci);assert saved['name']==cam['img_name'] and saved['split']==cam['split'];directory=OUT/scene/f'camera_{ci}';base=directory/'oracle';recapture=w.forward(mods['oracle'],ci);torch.cuda.synchronize();_,old_errors,C=decode(recapture,w.n,'oracle');an=anchor_view(recapture,w.n);mn,mx=rects(recapture,w.n);rad=recapture[3].cpu().numpy();flags,ids,pref=state(recapture,dict(n=w.n,w=w.cameras[ci]['w'],h=w.cameras[ci]['h']),'oracle')
            for key,arr in [('rgb',recapture[1].cpu().numpy()),('aux',recapture[2].cpu().numpy()),('radii',rad),('maxblend',recapture[9].cpu().numpy()),('contributors',ids),('double_coeff',C),('cache_errors',old_errors),('anchors',an),('rect_min',mn),('rect_max',mx)]:
                expected=data(base,key)
                if key in ['anchors','rect_min','rect_max']:arr,expected=arr[rad>0],expected[rad>0]
                assert arr.shape==expected.shape and arr.dtype==expected.dtype
                assert np.array_equal(np.ascontiguousarray(arr).view(np.uint8),np.ascontiguousarray(expected).view(np.uint8))
            vi=np.flatnonzero(rad>0);width,height=w.metadata['resolution'];B=np.maximum(np.abs(16*mn[vi].astype(float)-an[vi]),np.abs(np.minimum(16*mx[vi].astype(float),[width,height])-1-an[vi]));rank=np.array([int.from_bytes(hashlib.sha256(('v10cache/'+scene+'/'+cam['img_name']+'/'+str(i)).encode()).digest()[:8],'big') for i in vi],dtype=np.uint64);chosen=np.unique(np.r_[vi[np.argsort(rank,kind='stable')[:512]],vi[np.argsort(-(B[:,0]*B[:,1]),kind='stable')[:32]]]);cache=coarse_cache_audit(C,data(directory/'coarse','cache_errors'),an,mn,mx,rad,width,height,selected=chosen);comparisons={}
            for v in ['coarse','oracle_diag','coarse_diag']:
                folder=directory/v;ref=directory/v.removesuffix('_diag') if v.endswith('_diag') else base;keys=['rgb','aux','radii','maxblend','contributors','flags','double_coeff']+(['cache_errors'] if v.endswith('_diag') else []);bitwise={k:bool(np.array_equal(data(folder,k),data(ref,k))) for k in keys};numeric={}
                for loss in ['rgb','depth']:
                    numeric[loss]=[]
                    for j in range(6):
                        q=independent_group(data(folder,f'{loss}_gradient_{j}'),data(ref,f'{loss}_gradient_{j}'));other=saved['comparisons'][v]['VJP'][loss][j];assert q['gate']==other['gate'] and q['finite']==other['finite']
                        if q['finite']:
                            for key in ['reference_l2','relative','absolute_max']:assert math.isclose(q[key],other[key],rel_tol=1e-10,abs_tol=1e-15)
                        numeric[loss].append(q)
                zero=not data(folder,'flags').any() and not data(folder,'preflight').any();finite=all(np.isfinite(data(folder,k)).all() for k in ['rgb','aux','maxblend']);gate=bool(all(bitwise.values()) and zero and finite and all(q['gate'] for gs in numeric.values() for q in gs));assert gate==saved['comparisons'][v]['gate'];comparisons[v]=dict(bitwise=bitwise,VJP=numeric,gate=gate)
            oracle_finite=all(np.isfinite(data(base,f'{loss}_gradient_{j}')).all() for loss in ['rgb','depth'] for j in range(6));gate=bool(oracle_finite and all(q['gate'] for q in comparisons.values()));assert gate==saved['gate'];row=dict(scene=scene,camera=ci,name=cam['img_name'],split=cam['split'],oracle_recapture_bitwise=True,cache=cache,comparisons=comparisons,gate=gate);rows.append(row);(OUT/'audit_progress_v2.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(scene=scene,camera=ci,name=cam['img_name'],gate=gate,cache_edges=cache['checked_edges'])),flush=True);del recapture,C,old_errors,an,mn,mx,rad;gc.collect()
        assert w.input_hashes()==plan['model_inputs'][scene];del w;gc.collect();torch.cuda.empty_cache()
    gate=all(r['gate'] for r in rows);assert gate==real['all_gates'];out=dict(rows=rows,all_gates=gate,performance_measurement_allowed=gate,scene_count=2,camera_count=4,full_image_comparisons=True,full_VJP_independently_recomputed=True,oracle_recapture_bitwise=True,recapture_scope='All defined outputs; projection anchors/rects only where radii>0; original raw scratch retained.',audit_repair_plan_sha256=sha(OUT/'audit_repair_plan.json'),literal_bits_audited=True,script_sha256=sha(Path(__file__)),real_results_sha256=sha(OUT/'real_results.json'),plan_sha256=sha(OUT/'real_plan.json'),training=False,GT_quality_evaluated=False,scientific_goal_complete=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'audit_v2_failure.txt').write_text(traceback.format_exc());raise
