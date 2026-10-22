"""Fixed real RGB and weighted-depth VJPs with complete output/path comparison."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from benchmark_tgs_backward import ROOT,Workload
from probe_tgs_compositing_guard import modules,state
from probe_tgs_adaptive_window import decode
OUT=ROOT/'experiments/tgs_compositing_guard_v4'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'real_results.json';assert not target.exists();assert json.loads((OUT/'results.json').read_text())['all_gates']
    mods=modules();work=Workload();rgb_grad=work.grad.clone();depth_grad=torch.ones_like(work.grad_others[0])/(rgb_grad.shape[1]*rgb_grad.shape[2])
    plan=dict(workload=work.metadata,cameras=work.camera_metadata,script_sha256=sha(Path(__file__)),modules={v:sha(Path(m.__file__)) for v,m in mods.items()},
        depth_upstream='weighted depth channel0:1/(W*H), RGB and other channels zero',depth_upstream_sha256=hashlib.sha256(depth_grad.cpu().numpy().tobytes()).hexdigest(),
        required_flagged_pixels=[[0,116,742],[1,740,268]],training=False,timing=False,ground_truth_evaluation=False)
    (OUT/'real_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    for ci in range(2):
        outputs={};grads={};metrics={}
        for v,m in mods.items():
            res=work.forward(m,ci);torch.cuda.synchronize();counts,errors,coeff=decode(res,work.n)
            f=dict(n=work.n,w=work.cameras[ci]['w'],h=work.cameras[ci]['h']);flags,contrib,pref=state(res,f)
            a=dict(rgb=res[1].cpu().numpy(),aux=res[2].cpu().numpy(),radii=res[3].cpu().numpy(),maxblend=res[9].cpu().numpy(),flags=flags,contributors=contrib,counters=counts,preflight=pref)
            finite={};gg={}
            for loss in ['rgb','depth']:
                work.grad.copy_(rgb_grad if loss=='rgb' else torch.zeros_like(rgb_grad));work.grad_others.zero_()
                if loss=='depth':work.grad_others[0].copy_(depth_grad)
                g=work.backward(m,res,ci);torch.cuda.synchronize();ga=[x.cpu().numpy() for x in g];finite[loss]=all(np.isfinite(x).all() for x in ga);gg[loss]=ga
                a.update({f'{loss}_gradient_{i}':x for i,x in enumerate(ga)});del g
            metrics[v]=dict(all_gradients_finite=finite,tile_instances=int(res[0]),flagged_pixels=int((flags>0).sum()),median_flagged=int(((flags&1)>0).sum()),early_stop_flagged=int(((flags&2)>0).sum()),render_counts=counts.astype(np.uint64).sum(0).tolist(),preflight_counts=pref.astype(np.uint64).sum(0).tolist(),geometry_buffer_bytes=res[4].numel(),image_buffer_bytes=res[6].numel())
            outputs[v]=a;grads[v]=gg;np.savez_compressed(OUT/f'real_{ci}_{v}.npz',**a);del res,errors,coeff
        a,b=outputs['adaptive'],outputs['oracle64'];d=np.abs(a['rgb'].astype(float)-b['rgb'].astype(float));ad=np.abs(a['aux'].astype(float)-b['aux'].astype(float));md=float(np.abs(a['maxblend']-b['maxblend']).max())
        comp={}
        for loss in ['rgb','depth']:
            comp[loss]=[]
            for ga,gb in zip(grads['adaptive'][loss],grads['oracle64'][loss]):
                aa=ga.astype(float).ravel();bb=gb.astype(float).ravel();nr=float(np.linalg.norm(bb));err=float(np.linalg.norm(aa-bb)/max(nr,1e-30));absolute=float(np.abs(aa-bb).max());comp[loss].append(dict(relative=err,absolute_max=absolute,reference_l2=nr,gate=bool(err<=1e-3 if nr>1e-10 else absolute<=1e-6)))
        x,y=([116,742] if ci==0 else [740,268]);path=np.array_equal(a['contributors'],b['contributors']);radii=np.array_equal(a['radii'],b['radii']);finite=all(all(q['all_gradients_finite'].values()) for q in metrics.values())
        gate=float(d.max())<=3e-5 and float(ad.max())<=3e-5 and md<=3e-5 and path and radii and finite and flags is not None and int(a['flags'][y,x])>0 and all(q['gate'] for g in comp.values() for q in g)
        row=dict(camera=ci,image=work.camera_metadata[ci]['img_name'],metrics=metrics,rgb_max_abs=float(d.max()),rgb_mae=float(d.mean()),aux_per_channel_max=ad.reshape(7,-1).max(1).tolist(),maxblend_max_abs=md,contributor_median_identical=bool(path),radii_identical=bool(radii),outlier_pixel=[x,y],outlier_flag=int(a['flags'][y,x]),outlier_depth=[float(a['aux'][0,y,x]),float(b['aux'][0,y,x])],gradient_comparisons=comp,gate=bool(gate))
        rows.append(row);print(json.dumps(row),flush=True);del outputs,grads
    target.write_text(json.dumps(dict(plan_sha256=sha(OUT/'real_plan.json'),rows=rows,all_gates=all(r['gate'] for r in rows),training=False,timing=False,quality_gain_established=False),indent=2)+'\n')
if __name__=='__main__':main()
