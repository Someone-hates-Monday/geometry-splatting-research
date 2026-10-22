"""No-grad matched-query diagnostic of two final models after failed quality gate."""
from pathlib import Path
import argparse,hashlib,json,math,os,tempfile,time,traceback,zipfile
import numpy as np
import torch
from research.history_gap_live_adapter import LiveAdapter,load_native
from research.benchmark_median_cuda import load_config
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);out=parser.parse_args().out.resolve()
    assert not out.exists();out.mkdir(parents=True)
    pilot=R/'experiments/history_gap_pilot_v1';audit=read(R/'results/history_gap_pilot_audit_v1.json')
    assert audit['audit_gate_pass'] and not audit['pilot_gate_pass']
    pm=read(pilot/'manifest.json');assert pm['status']=='complete' and not pm['gate_pass']
    sources=[Path(__file__),R/'HISTORY_GAP_POSTHOC_PROTOCOL.md',R/'history_gap_live_adapter.py',R/'history_gap_gpu.py',R/'history_gap_gpu_v2.py',
        R/'benchmark_median_cuda.py',ROOT/'scene/gaussian_model.py',ROOT/'scene/__init__.py',ROOT/'utils/camera_utils.py',ROOT/'arguments/__init__.py']
    m=dict(status='running',pid=os.getpid(),epoch=time.time(),pilot_manifest_sha256=sha(pilot/'manifest.json'),
        pilot_audit_sha256=sha(R/'results/history_gap_pilot_audit_v1.json'),sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources},model_inputs={})
    save(out/'manifest.json',m)
    try:
        from scene import GaussianModel,Scene
        torch.set_num_threads(1)
        rec=read(pilot/'history_gap/history_gap_training_integration.json');qm=rec['binaries']['query'];cm=rec['binaries']['csr'];wm=rec['binaries']['loss']
        adapter=LiveAdapter(load_native(qm['binary_path'],qm['binary_sha256']),load_native(ROOT/cm['binary_relative_path'],cm['binary_sha256']),load_native(ROOT/wm['path'],wm['sha256']))
        m['binaries']=rec['binaries'];all_arrays={};per_camera=[]
        for label in ['control','history_gap']:
            model=pilot/label;ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=pm['artifacts_sha256'][ply.relative_to(pilot).as_posix()]
            assert sha(ply)==expected;m['model_inputs'][label]=dict(ply_sha256=expected,cfg_sha256=sha(model/'cfg_args'))
            ds,pipe=load_config(model)
            with tempfile.TemporaryDirectory(prefix='gap-mechanism-',dir=out) as temp:
                scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
                gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
                cameras=scene.getTrainCameras();assert [c.image_name for c in cameras]==pm['train_names']
                bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda');rows=[]
                with torch.no_grad():
                    for cam in cameras:
                        pixels=np.random.default_rng(np.random.SeedSequence([13024,int(cam.image_name)])).choice(cam.image_width*cam.image_height,512,replace=False)
                        cap=adapter.capture(cam,gs,pipe,bg,ds.kernel_size);loss,detail=adapter.loss(gs,cap,pixels)
                        active=cap.counts[torch.tensor(pixels,device='cuda')].cpu().numpy()>0
                        values=np.full(512,math.log(4)**2,dtype=np.float64);info=np.full((512,8),np.nan,dtype=np.float64)
                        values[active]=detail['values'].cpu().numpy();info[active]=detail['info'].cpu().numpy()
                        counts=cap.counts[torch.tensor(pixels,device='cuda')].cpu().numpy()
                        assert abs(float(loss)-float(values.mean()))<1e-12 and np.isfinite(values).all()
                        dest=out/f'{label}_{cam.image_name}.npz'
                        np.savez_compressed(dest,pixels=pixels,active=active,values=values,info=info,counts=counts)
                        row=dict(label=label,camera=cam.image_name,loss=float(loss),empty=int((~active).sum()),
                            nonempty_mean=float(values[active].mean()) if active.any() else None,
                            native_status_counts={str(int(k)):int(v) for k,v in zip(*np.unique(info[active,0],return_counts=True))})
                        per_camera.append(row);rows.append((pixels,active,values,info));save(out/'progress.json',row)
                        print(f'{label} {cam.image_name} loss={row["loss"]:.6g} empty={row["empty"]}',flush=True)
                        del cap,detail
                all_arrays[label]=rows
                assert sha(ply)==expected
                del gs,scene,cameras;torch.cuda.empty_cache()
        combined={};data={}
        for label,rows in all_arrays.items():
            active=np.concatenate([x[1] for x in rows]);values=np.concatenate([x[2] for x in rows]);info=np.concatenate([x[3] for x in rows]);data[label]=(active,values,info)
            combined[label]=dict(samples=len(values),empty_fraction=float((~active).mean()),full_mean=float(values.mean()),
                empty_constant_contribution=float((~active).mean()*math.log(4)**2),
                nonempty_conditional_mean=float(values[active].mean()),nonempty_full_denominator_contribution=float(values[active].sum()/len(values)),
                native_status_counts={str(int(k)):int(v) for k,v in zip(*np.unique(info[active,0],return_counts=True))})
            assert abs(combined[label]['full_mean']-combined[label]['empty_constant_contribution']-combined[label]['nonempty_full_denominator_contribution'])<1e-12
        assert all(np.array_equal(a[0],b[0]) for a,b in zip(all_arrays['control'],all_arrays['history_gap']))
        a,va,_=data['control'];b,vb,_=data['history_gap'];common=a&b
        paired=dict(common_nonempty=int(common.sum()),control_mean=float(va[common].mean()),history_gap_mean=float(vb[common].mean()),
            mean_delta=float((vb[common]-va[common]).mean()),full_mean_delta=float((vb-va).mean()),
            empty_to_nonempty=int((~a&b).sum()),nonempty_to_empty=int((a&~b).sum()),both_empty=int((~a&~b).sum()))
        summary=dict(status='complete',combined=combined,paired=paired,per_camera=per_camera,
            scope='Post-hoc no-grad512 matched pixels on all42 train views of each final model. Not a replacement quality gate, causal explanation, significance test or new training.')
        save(out/'summary.json',summary)
        for name,h in m['sources'].items():assert sha(ROOT/name)==h
        with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
            for name in m['sources']:z.write(ROOT/name,name)
        m.update(status='complete',artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']})
        save(out/'manifest.json',m);print(json.dumps(dict(combined=combined,paired=paired)),flush=True)
    except BaseException as exc:
        m.update(status='failed',failure=repr(exc));save(out/'manifest.json',m);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
