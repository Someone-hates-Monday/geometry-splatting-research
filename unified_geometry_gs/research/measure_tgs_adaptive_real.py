"""Two complete fixed real cameras: no GT, optimizer, timing or training."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import torch
from benchmark_tgs_backward import ROOT,Workload
from probe_tgs_adaptive_window import decode
OUT=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    target=OUT/'real_results.json';assert not target.exists()
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_candidate_gates'] and not audit['fd_support_failures']
    mods={}
    for v in ['adaptive','oracle64']:
        r=json.loads((OUT/v/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256']
        spec=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[v]=m
    work=Workload();sigma=work.sigma.cpu().numpy().ravel()
    plan=dict(workload=work.metadata,cameras=work.camera_metadata,variants=list(mods),modules={v:sha(Path(m.__file__)) for v,m in mods.items()},script_sha256=sha(Path(__file__)),
        fixed_before_outputs=True,training=False,timing=False,ground_truth_evaluation=False,
        gates=dict(rgb_max_abs_to_oracle=3e-5,geometry_scalar_SH_vjp_relative_to_oracle=1e-3,all_gradients_finite=True,radii_identical=True),
        sigma_quantiles_levels=[0,.001,.01,.1,.5,.9,.99,.999,1],sigma_quantiles=np.quantile(sigma,[0,.001,.01,.1,.5,.9,.99,.999,1]).tolist(),sigma_below_one=int((sigma<1).sum()),
        limitations='Oracle uses double window arithmetic with float gradient accumulation, not independent true derivatives of every real triangle. No image-quality/training/latency claim.')
    (OUT/'real_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    for ci in range(2):
        saved={};gradients={};metrics={}
        for v,m in mods.items():
            res=work.forward(m,ci);g=work.backward(m,res,ci);torch.cuda.synchronize()
            rgb=res[1].cpu().numpy();aux=res[2].cpu().numpy();radii=res[3].cpu().numpy();counts,errors,coeffs=decode(res,work.n)
            ga=[x.detach().cpu().numpy() for x in g]
            saved[v]=dict(rgb=rgb,aux=aux,radii=radii,counters=counts);gradients[v]=ga
            metrics[v]=dict(tile_instances=int(res[0]),visible=int((radii>0).sum()),counters=counts.astype(np.uint64).sum(0).tolist(),geometry_buffer_bytes=res[4].numel(),all_gradients_finite=all(np.isfinite(a).all() for a in ga),nonfinite_gradient_counts=[int((~np.isfinite(a)).sum()) for a in ga])
            np.savez_compressed(OUT/f'real_{ci}_{v}.npz',**saved[v],**{f'gradient_{i}':a for i,a in enumerate(ga)})
            del res,g,errors,coeffs
        d=np.abs(saved['adaptive']['rgb'].astype(float)-saved['oracle64']['rgb'].astype(float));ad=np.abs(saved['adaptive']['aux'].astype(float)-saved['oracle64']['aux'].astype(float))
        comparisons=[]
        for a,b in zip(gradients['adaptive'],gradients['oracle64']):
            aa=a.astype(float).ravel();bb=b.astype(float).ravel();norm=float(np.linalg.norm(bb))
            comparisons.append(dict(relative=float(np.linalg.norm(aa-bb)/max(norm,1e-30)),absolute_max=float(np.abs(aa-bb).max()),reference_l2=norm))
        # Return slots: geometry,sigma,color,opacity,SH,means2D. Color is empty for SH.
        required=[i for i in [0,1,3,4] if gradients['oracle64'][i].size]
        radii_equal=np.array_equal(saved['adaptive']['radii'],saved['oracle64']['radii'])
        gate=d.max()<=3e-5 and radii_equal and all(q['all_gradients_finite'] for q in metrics.values()) and all(comparisons[i]['relative']<=1e-3 for i in required)
        row=dict(camera=ci,image=work.camera_metadata[ci]['img_name'],metrics=metrics,rgb_max_abs=float(d.max()),rgb_mae=float(d.mean()),pixels_rgb_delta_gt_3e5=int((d.max(0)>3e-5).sum()),aux_max_abs=float(ad.max()),radii_identical=radii_equal,gradient_comparisons=comparisons,gate=bool(gate))
        rows.append(row);print(json.dumps(row),flush=True)
        del saved,gradients
    target.write_text(json.dumps(dict(plan_sha256=sha(OUT/'real_plan.json'),rows=rows,all_gates=all(r['gate'] for r in rows),training=False,timing=False,quality_gain_established=False),indent=2)+'\n')
if __name__=='__main__':main()
