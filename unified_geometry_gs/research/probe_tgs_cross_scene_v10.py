"""Four complete new camera inputs, frozen modules, full saved VJPs; no optimizer."""
import gc,json,math,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,OUT,V9,VARIANTS,sha,modules,CrossWorkload,decode,state
from audit_tgs_compensated_baseline_diagnostic import anchor_view,rects

FORWARD=['rgb','aux','radii','maxblend','contributors']
PARITY=FORWARD+['flags','double_coeff','cache_errors']

def save(folder,key,value):
    target=folder/(key+'.npy');assert not target.exists();np.save(target,np.asarray(value),allow_pickle=False)
def load(folder,key):return np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
def equal(a,b,key):return bool(np.array_equal(load(a,key),load(b,key)))

def group(a,b):
    a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1);assert a.shape==b.shape;aa=[];bb=[];dd=[];maximum=0.;finite=True
    for start in range(0,len(a),1048576):
        x=np.array(a[start:start+1048576],dtype=float);y=np.array(b[start:start+1048576],dtype=float);finite &= bool(np.isfinite(x).all() and np.isfinite(y).all());d=x-y;maximum=max(maximum,float(np.abs(d).max(initial=0)));aa.append(float(x@x));bb.append(float(y@y));dd.append(float(d@d))
    if not finite:return dict(finite=False,gate=False,relative=None,absolute_max=None,reference_l2=None)
    reference=math.sqrt(math.fsum(bb));error=math.sqrt(math.fsum(dd));rel=error/max(reference,1e-30);return dict(finite=True,relative=rel,absolute_max=maximum,reference_l2=reference,native_l2=math.sqrt(math.fsum(aa)),error_l2=error,gate=bool(rel<=1e-5 if reference>1e-10 else maximum<=1e-6))

def compare_camera(directory):
    base=directory/'oracle';outcomes={}
    for v in ['coarse','oracle_diag','coarse_diag']:
        folder=directory/v;ref=directory/v.removesuffix('_diag') if v.endswith('_diag') else base;keys=PARITY if v.endswith('_diag') else FORWARD+['double_coeff'];matches={k:equal(folder,ref,k) for k in keys};groups={loss:[group(load(folder,f'{loss}_gradient_{j}'),load(ref,f'{loss}_gradient_{j}')) for j in range(6)] for loss in ['rgb','depth']};finite_forward=all(np.isfinite(load(folder,k)).all() for k in ['rgb','aux','maxblend']);zero=not load(folder,'flags').any() and not load(folder,'preflight').any()
        outcomes[v]=dict(reference=ref.name,bitwise=matches,VJP=groups,finite_forward=bool(finite_forward),zero_replay_and_preflight=bool(zero),gate=bool(all(matches.values()) and finite_forward and zero and all(g['gate'] for gs in groups.values() for g in gs)))
    oracle_finite=all(np.isfinite(load(base,f'{loss}_gradient_{j}')).all() for loss in ['rgb','depth'] for j in range(6));return dict(comparisons=outcomes,oracle_gradients_finite=bool(oracle_finite),gate=bool(oracle_finite and all(x['gate'] for x in outcomes.values())))

def main():
    audit=json.loads((OUT/'input_audit.json').read_text());assert audit['passed'];target=OUT/'real_results.json';assert not target.exists() and not (OUT/'real_plan.json').exists();mods=modules();scenes=[r['scene'] for r in audit['scenes']];inputs={};metadata={}
    for scene in scenes:
        w=CrossWorkload(scene);inputs[scene]=w.input_hashes();metadata[scene]=w.metadata;del w;gc.collect();torch.cuda.empty_cache()
    scripts=['tgs_cross_scene_common_v10.py','audit_tgs_cross_scene_inputs_v10.py','probe_tgs_cross_scene_v10.py','audit_tgs_cross_scene_precision_v10.py']
    plan=dict(input_audit_sha256=sha(OUT/'input_audit.json'),protocol_sha256=sha(ROOT/'TGS_CROSS_SCENE_PROTOCOL_V10.md'),scripts={s:sha(ROOT/s) for s in scripts},model_inputs=inputs,metadata=metadata,modules={v:dict(path=m.__file__,sha256=sha(Path(m.__file__))) for v,m in mods.items()},training=False,timing=False)
    (OUT/'real_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    for scene in scenes:
        work=CrossWorkload(scene);assert work.input_hashes()==inputs[scene]
        for ci,cam in enumerate(work.camera_metadata):
            directory=OUT/scene/f'camera_{ci}';assert not directory.exists();directory.mkdir(parents=True);f=dict(n=work.n,w=work.cameras[ci]['w'],h=work.cameras[ci]['h']);meta={}
            for v,m in mods.items():
                folder=directory/v;folder.mkdir();r=work.forward(m,ci);torch.cuda.synchronize();ct,errors,coeff=decode(r,work.n,v);flags,ids,pf=state(r,f,v)
                arrays=dict(rgb=r[1].cpu().numpy(),aux=r[2].cpu().numpy(),radii=r[3].cpu().numpy(),maxblend=r[9].cpu().numpy(),flags=flags,contributors=ids,counters=ct,preflight=pf,double_coeff=coeff,cache_errors=errors)
                if v=='oracle':
                    mn,mx=rects(r,work.n);arrays.update(anchors=anchor_view(r,work.n),rect_min=mn.copy(),rect_max=mx.copy())
                for key,value in arrays.items():save(folder,key,value)
                meta[v]=dict(tiles=int(r[0]),geometry_buffer_bytes=r[4].numel(),image_buffer_bytes=r[6].numel(),render_counts=ct.astype(np.uint64).sum(0).tolist(),preflight_counts=pf.astype(np.uint64).sum(0).tolist())
                del arrays,ct,errors,coeff,flags,ids,pf
                for loss in ['rgb','depth']:
                    work.set_loss(loss);gradients=work.backward(m,r,ci);torch.cuda.synchronize()
                    for j,g in enumerate(gradients):save(folder,f'{loss}_gradient_{j}',g.cpu().numpy())
                    del gradients;print(json.dumps(dict(scene=scene,camera=ci,name=cam['img_name'],variant=v,loss=loss,saved=True)),flush=True)
                del r
            q=compare_camera(directory);row=dict(scene=scene,camera=ci,name=cam['img_name'],split=cam['split'],resolution=[f['w'],f['h']],triangles=work.n,metrics=meta,**q);rows.append(row);(OUT/'real_progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(scene=scene,camera=ci,gate=q['gate'],max_relative=max(g['relative'] or 0 for cmp in q['comparisons'].values() for gs in cmp['VJP'].values() for g in gs))),flush=True)
        assert work.input_hashes()==inputs[scene];del work;gc.collect();torch.cuda.empty_cache()
    target.write_text(json.dumps(dict(rows=rows,all_gates=all(r['gate'] for r in rows),model_inputs_unchanged=True,plan_sha256=sha(OUT/'real_plan.json'),script_sha256=sha(Path(__file__)),training=False,timing=False),indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'real_failure.txt').write_text(traceback.format_exc());raise
