"""Four full cameras, counter-removal parity, lean RGB/depth VJP validation."""
import json
from pathlib import Path
import numpy as np
import torch
from tgs_compensated_common import ROOT,OUT,VARIANTS,sha,modules,workload,v5_oracle
from tgs_compensated_common import state
from tgs_compensated_common import decode
from probe_tgs_radius_contraction import check_group

def main():
    target=OUT/'real_results.json';assert not target.exists();assert json.loads((OUT/'case_results.json').read_text())['all_gates']
    mods=modules();work=workload();oldmod=v5_oracle();rgb_grad=work.grad.clone();depth=torch.ones_like(work.grad_others[0])/(rgb_grad.shape[1]*rgb_grad.shape[2])
    plan=dict(workload=work.metadata,cameras=work.camera_metadata,script_sha256=sha(Path(__file__)),modules={v:sha(Path(m.__file__)) for v,m in mods.items()},depth_upstream='channel0=1/(W*H), other/RGB zero',depth_upstream_sha256=__import__('hashlib').sha256(depth.cpu().numpy().tobytes()).hexdigest(),training=False,timing=False)
    (OUT/'real_plan.json').write_text(json.dumps(plan,indent=2)+'\n');rows=[]
    for ci in range(len(work.cameras)):
        data={};meta={}
        for v,m in mods.items():
            r=work.forward(m,ci);torch.cuda.synchronize();counts,err,coeff=decode(r,work.n);f=dict(n=work.n,w=work.cameras[ci]['w'],h=work.cameras[ci]['h']);flags,contrib,pref=state(r,f)
            a=dict(rgb=r[1].cpu().numpy(),aux=r[2].cpu().numpy(),radii=r[3].cpu().numpy(),maxblend=r[9].cpu().numpy(),flags=flags,contributors=contrib,counters=counts,preflight=pref,double_coeff=coeff,cache_errors=err)
            finite={}
            if not v.endswith('_diag'):
                assert not counts.any() and not pref.any()
                for loss in ['rgb','depth']:
                    work.grad.copy_(rgb_grad if loss=='rgb' else torch.zeros_like(rgb_grad));work.grad_others.zero_()
                    if loss=='depth':work.grad_others[0].copy_(depth)
                    g=work.backward(m,r,ci);torch.cuda.synchronize();ga=[x.cpu().numpy() for x in g];finite[loss]=all(np.isfinite(x).all() for x in ga)
                    a.update({f'{loss}_gradient_{i}':x for i,x in enumerate(ga)});del g
            meta[v]=dict(all_gradients_finite=finite,tiles=int(r[0]),flagged_pixels=int((flags>0).sum()),geometry_buffer_bytes=r[4].numel(),image_buffer_bytes=r[6].numel(),render_counts=counts.astype(np.uint64).sum(0).tolist(),preflight_counts=pref.astype(np.uint64).sum(0).tolist())
            data[v]=a;np.savez_compressed(OUT/f'real_{ci}_{v}.npz',**a);del r
        parity={}
        for v in ['guard','oracle']:
            parity[v]={k:bool(np.array_equal(data[v][k],data[v+'_diag'][k])) for k in ['rgb','aux','radii','maxblend','flags','contributors','double_coeff','cache_errors']}
        a,b=data['guard'],data['oracle'];rgb=float(np.abs(a['rgb']-b['rgb']).max());aux=np.abs(a['aux'].astype(float)-b['aux'].astype(float)).reshape(7,-1).max(1);mb=float(np.abs(a['maxblend']-b['maxblend']).max());ids=np.array_equal(a['contributors'],b['contributors']);rad=np.array_equal(a['radii'],b['radii']);flags=a['flags']>0
        flagged_equal=all(np.array_equal(a[k][:,flags],b[k][:,flags]) for k in ['rgb','aux','contributors']);comp={loss:[check_group(a[f'{loss}_gradient_{i}'].astype(float).ravel(),b[f'{loss}_gradient_{i}'].astype(float).ravel()) for i in range(6)] for loss in ['rgb','depth']}
        oldres=work.forward(oldmod,ci);torch.cuda.synchronize()
        old=dict(rgb=float(np.abs(oldres[1].cpu().numpy()-b['rgb']).max()),aux=np.abs(oldres[2].cpu().numpy()-b['aux']).reshape(7,-1).max(1).tolist(),radii_equal=bool(np.array_equal(oldres[3].cpu().numpy(),b['radii'])),maxblend=float(np.abs(oldres[9].cpu().numpy()-b['maxblend']).max()))
        assert old['rgb']==0 and max(old['aux'])==0 and old['maxblend']==0 and old['radii_equal']
        del oldres
        regression=None
        if ci<2:
            x,y=([116,742] if ci==0 else [740,268]);regression=dict(pixel=[x,y],flag=int(a['flags'][y,x]),RGB_abs=float(np.abs(a['rgb'][:,y,x]-b['rgb'][:,y,x]).max()),aux_abs=np.abs(a['aux'][:,y,x]-b['aux'][:,y,x]).tolist(),paths_identical=bool(np.array_equal(a['contributors'][:,y,x],b['contributors'][:,y,x])))
            assert regression['paths_identical'] and max(regression['aux_abs'])<=3e-5
        finite=all(all(meta[v]['all_gradients_finite'].values()) for v in ['guard','oracle'])
        gate=all(all(x.values()) for x in parity.values()) and rgb<=3e-5 and aux.max()<=3e-5 and mb<=3e-5 and ids and rad and flagged_equal and finite and all(q['gate'] for gg in comp.values() for q in gg)
        row=dict(camera=ci,image=work.camera_metadata[ci]['img_name'],split=work.camera_metadata[ci]['split'],metrics=meta,instrumentation_parity=parity,RGB_abs=rgb,aux_per_channel_abs=aux.tolist(),maxblend_abs=mb,contributors_identical=bool(ids),radii_identical=bool(rad),flagged_bitwise_equal=flagged_equal,gradient_comparisons=comp,old_v5_oracle_comparison=old,regression_pixel=regression,gate=bool(gate))
        rows.append(row);print(json.dumps(row),flush=True);del data
    target.write_text(json.dumps(dict(plan_sha256=sha(OUT/'real_plan.json'),rows=rows,all_gates=all(r['gate'] for r in rows),training=False,timing=False),indent=2)+'\n')
if __name__=='__main__':main()
