"""Seven full cameras, two independently gated candidates against original native64."""
import hashlib,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_window_candidates_common import ROOT,OUT,BASES,VARIANTS,sha,modules,workload,decode,state,strict_groups
from probe_tgs_radius_contraction import check_group
def thash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def main():
    assert not (OUT/'real_results.json').exists();case=json.loads((OUT/'case_results.json').read_text());assert case['candidate_gates']['oracle'];mods=modules();work=workload();rgb_up=work.grad.clone();depth_up=torch.ones_like(work.grad_others[0])/(rgb_up.shape[1]*rgb_up.shape[2]);inputs={k:thash(getattr(work,k)) for k in ['points','sigma','opacity','sh','counts','offsets']}
    p=json.loads((OUT/'input_plan.json').read_text());badplan=json.loads((ROOT/'experiments/tgs_compensated_renderer_v8/depth_failure_plan.json').read_text());bad={r['name']:r['bad'] for r in badplan['rows']}
    plan=dict(input_plan_sha256=sha(OUT/'input_plan.json'),script_sha256=sha(Path(__file__)),workload=work.metadata,cameras=work.camera_metadata,model_input_hashes=inputs,modules={v:sha(Path(m.__file__)) for v,m in mods.items()},depth_upstream_sha256=thash(depth_up),training=False,timing=False);(OUT/'real_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    for ci,cam in enumerate(work.camera_metadata):
        data={};meta={};f=dict(n=work.n,w=work.cameras[ci]['w'],h=work.cameras[ci]['h'])
        for v,m in mods.items():
            r=work.forward(m,ci);torch.cuda.synchronize();ct,e,c=decode(r,work.n,v);flag,ids,pf=state(r,f,v);a=dict(rgb=r[1].cpu().numpy(),aux=r[2].cpu().numpy(),radii=r[3].cpu().numpy(),maxblend=r[9].cpu().numpy(),flags=flag,contributors=ids,counters=ct,preflight=pf,double_coeff=c,cache_errors=e)
            finite={}
            if not v.endswith('_diag'):
                assert not ct.any() and not pf.any()
                for loss in ['rgb','depth']:
                    work.grad.copy_(rgb_up if loss=='rgb' else torch.zeros_like(rgb_up));work.grad_others.zero_()
                    if loss=='depth':work.grad_others[0].copy_(depth_up)
                    gg=work.backward(m,r,ci);torch.cuda.synchronize();vals=[t.cpu().numpy() for t in gg];finite[loss]=all(np.isfinite(t).all() for t in vals);a.update({f'{loss}_gradient_{j}':t for j,t in enumerate(vals)});del gg
            meta[v]=dict(all_gradients_finite=finite,tiles=int(r[0]),flagged_pixels=int((flag>0).sum()),geometry_buffer_bytes=r[4].numel(),image_buffer_bytes=r[6].numel(),render_counts=ct.astype(np.uint64).sum(0).tolist(),preflight_counts=pf.astype(np.uint64).sum(0).tolist());data[v]=a;np.savez_compressed(OUT/f'real_{ci}_{v}.npz',**a);del r
        b=data['oracle'];parity={}
        for v in BASES:parity[v]={k:bool(np.array_equal(data[v][k],data[v+'_diag'][k])) for k in ['rgb','aux','radii','maxblend','flags','contributors','double_coeff','cache_errors']}
        outcomes={}
        for v in ['coarse','standard']:
            a=data[v];rgb=float(np.abs(a['rgb']-b['rgb']).max());aux=np.abs(a['aux'].astype(float)-b['aux'].astype(float)).reshape(7,-1).max(1);mb=float(np.abs(a['maxblend']-b['maxblend']).max());paths=bool(np.array_equal(a['contributors'],b['contributors']) and np.array_equal(a['radii'],b['radii']));flags=a['flags']>0;flagged=all(np.array_equal(a[k][:,flags],b[k][:,flags]) for k in ['rgb','aux','contributors']);groups={loss:[check_group(a[f'{loss}_gradient_{j}'].astype(float).ravel(),b[f'{loss}_gradient_{j}'].astype(float).ravel()) for j in range(6)] for loss in ['rgb','depth']}
            finite=all(meta[v]['all_gradients_finite'].values());vf=all(g['gate'] for gs in groups.values() for g in gs);exact=all(np.array_equal(a[k],b[k]) for k in ['rgb','aux','radii','maxblend','contributors']);zero_replay=not a['flags'].any() and not data[v+'_diag']['preflight'].any();reg=[]
            for pt in bad.get(cam['img_name'],[]):
                x,y=pt['pixel'];reg.append(dict(pixel=[x,y],depth_abs=float(abs(float(a['aux'][0,y,x])-float(b['aux'][0,y,x]))),paths_equal=bool(np.array_equal(a['contributors'][:,y,x],b['contributors'][:,y,x])),flag=int(a['flags'][y,x])))
            gate=bool(all(parity[v].values()) and all(parity['oracle'].values()) and max(rgb,float(aux.max()),mb)<=3e-5 and paths and flagged and finite and vf and all(t['depth_abs']<=3e-5 and t['paths_equal'] for t in reg))
            if v=='coarse':gate &= exact and zero_replay and all(strict_groups(gs) for gs in groups.values())
            outcomes[v]=dict(RGB_abs=rgb,aux_per_channel_abs=aux.tolist(),maxblend_abs=mb,paths_radii_identical=paths,flagged_endpoints_bitwise=bool(flagged),all_forward_bitwise=exact,zero_replay=bool(zero_replay),gradient_groups=groups,all_gradients_finite=finite,regression_14_pixels=reg,gate=bool(gate))
        row=dict(camera=ci,name=cam['img_name'],split=cam['split'],new_holdout=bool(cam.get('heldout_for_v9',False)),metrics=meta,instrumentation_parity=parity,candidates=outcomes);rows.append(row);(OUT/'real_progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(camera=ci,name=cam['img_name'],outcomes={v:dict(gate=q['gate'],RGB_abs=q['RGB_abs'],aux_abs=q['aux_per_channel_abs'],all_forward_bitwise=q['all_forward_bitwise'],max_VJP=max(g['relative'] for gs in q['gradient_groups'].values() for g in gs)) for v,q in outcomes.items()})),flush=True);del data
    assert inputs=={k:thash(getattr(work,k)) for k in inputs};gates={v:all(r['candidates'][v]['gate'] for r in rows) for v in ['coarse','standard']};(OUT/'real_results.json').write_text(json.dumps(dict(rows=rows,candidate_gates=gates,all_gates=all(gates.values()),model_inputs_unchanged=True,plan_sha256=sha(OUT/'real_plan.json'),training=False,timing=False),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'real_execution_failure.txt').write_text(traceback.format_exc());raise
