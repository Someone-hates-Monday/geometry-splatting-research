"""Canonical precision and exact forward instrumentation parity on all127 inputs."""
import json,time,traceback
from pathlib import Path
import numpy as np
from tgs_phase_common import ROOT,OUT,VARIANTS,sha,modules
from probe_tgs_compositing_guard import state
from probe_tgs_stable_window import SafeNative,SafeAuxNative,signature
from probe_tgs_adaptive_window import decode
from probe_tgs_radius_contraction import check_group
OLD=ROOT/'experiments/tgs_compositing_guard_v4';REF=ROOT/'experiments/tgs_adaptive_window_v3'

def groups(a,b,n,degree):
    return {key:check_group(a[sl],b[sl]) for key,sl in [('geometry',slice(0,9*n)),('sigma_opacity',slice(9*n,11*n))]+([('SH',slice(11*n,None))] if degree>=0 else [])}

def main():
    assert not (OUT/'case_results.json').exists() and not (OUT/'case_progress.json').exists()
    plan=json.loads((OUT/'input_plan.json').read_text());assert plan['protocol_sha256']==sha(ROOT/'TGS_PHASE_COST_PROTOCOL_V5.md');mods=modules();rows=[];started=time.time()
    execution=dict(input_plan_sha256=sha(OUT/'input_plan.json'),script_sha256=sha(Path(__file__)),modules={v:sha(Path(m.__file__)) for v,m in mods.items()},training=False)
    (OUT/'case_execution_plan.json').write_text(json.dumps(execution,indent=2)+'\n')
    for ci,e in enumerate(plan['cases']):
        path=ROOT/e['input'];assert sha(path)==e['sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        new=e['family']=='compositing_cases';rp=OLD/('cpu_reference_'+f['name']+'.npz') if new else REF/('reference_'+f['name']+'.npz')
        with np.load(rp) as z:ref={k:z[k].copy() for k in z.files}
        patterns=f.get('patterns',np.array([f['pixel_grad']]));data={};case=[]
        for v,m in mods.items():
            f['pixel_grad']=patterns[0];native=(SafeAuxNative if f['kind']=='aux' else SafeNative)(m,f);forward=native.forward(f['theta']);sig=signature(forward[1],f);counts,errors,coeff=decode(forward[1],f['n']);flags,contrib,pref=state(forward[1],f);grads=[];prs=[]
            for pi,p in enumerate(patterns):
                f['pixel_grad']=p;native.grad.zero_()
                for (x,y),weight in zip(f['pixels'],p):native.grad[:,y,x]=native.t(weight)
                g,_=native.backward(forward);grads.append(g);cg=groups(g,ref['gradients'][pi],f['n'],f['degree'])
                vg=abs(float(g@f['special_direction']))<=1e-6 if f['check']=='homothety_nullspace' else all(q['gate'] for q in cg.values())
                fe=float(np.abs(forward[-1]-ref['samples']).max());prs.append(dict(pattern=pi,CPU_groups=cg,CPU_forward_abs=fe,CPU_gate=bool(vg and fe<=3e-5)))
            r=forward[1];a=dict(native_rgb=r[1].cpu().numpy(),native_aux=r[2].cpu().numpy(),native_radii=r[3].cpu().numpy(),maxblend=r[9].cpu().numpy(),native_samples=forward[-1],native_gradients=np.array(grads),flags=flags,contributors=contrib,counters=counts,preflight=pref,double_coeff=coeff,cache_errors=errors,rect_min=np.array(sig['rect_min']),rect_max=np.array(sig['rect_max']))
            assert all(np.isfinite(a[k]).all() for k in ['native_rgb','native_aux','native_gradients'])
            if not v.endswith('_diag'):assert not counts.any() and not pref.any()
            data[v]=a;full=float(np.abs(a['native_rgb']-ref['rgb']).max());case.append(dict(name=f['name'],family=e['family'],variant=v,patterns=prs,CPU_full_rgb_abs=full,base_signature=sig,CPU_gate=bool(all(p['CPU_gate'] for p in prs) and full<=3e-5)))
            np.savez_compressed(OUT/(v+'_'+f['name']+'.npz'),**a)
        oracle=data['oracle'];nativechecks={};parities={}
        for v,a in data.items():
            delta={k:float(np.abs(a[k].astype(float)-oracle[k].astype(float)).max()) for k in ['native_rgb','native_aux','maxblend']}
            ids=np.array_equal(a['contributors'],oracle['contributors']) and np.array_equal(a['native_radii'],oracle['native_radii'])
            gg=[groups(g,r,f['n'],f['degree']) for g,r in zip(a['native_gradients'],oracle['native_gradients'])]
            nativechecks[v]=dict(deltas=delta,paths_radii_equal=bool(ids),gradient_groups=gg,gate=bool(max(delta.values())<=3e-5 and ids and all(q['gate'] for g in gg for q in g.values())))
        for v in ['guard','oracle']:
            a,b=data[v],data[v+'_diag'];equal={k:bool(np.array_equal(a[k],b[k])) for k in ['native_rgb','native_aux','native_radii','maxblend','flags','contributors','double_coeff','cache_errors']}
            gg=[groups(g,r,f['n'],f['degree']) for g,r in zip(a['native_gradients'],b['native_gradients'])]
            small=all(q['relative']<=1e-5 if q['reference_l2']>1e-10 else q['absolute_max']<=1e-6 for g in gg for q in g.values())
            parities[v]=dict(forward_bitwise=equal,VJP_groups=gg,VJP_small_difference=bool(small),gate=bool(all(equal.values()) and small))
        for row in case:
            v=row['variant'];base=v.removesuffix('_diag');row.update(native=nativechecks[v],instrumentation_parity=parities[base],gate=bool(nativechecks[v]['gate'] and parities[base]['gate'] and (new or row['CPU_gate'])))
        rows.extend(case);(OUT/'case_progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(dict(case=ci+1,total=len(plan['cases']),name=f['name'],all_gates=all(r['gate'] for r in case),parity={v:p['gate'] for v,p in parities.items()})),flush=True)
    result=dict(records=rows,cases=len(plan['cases']),VJPs=sum(len(r['patterns']) for r in rows),all_gates=all(r['gate'] for r in rows),failed={v:[r['name'] for r in rows if r['variant']==v and not r['gate']] for v in mods},input_plan_sha256=sha(OUT/'input_plan.json'),execution_plan_sha256=sha(OUT/'case_execution_plan.json'),seconds=time.time()-started,training=False)
    (OUT/'case_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'case_execution_failure.txt').write_text(traceback.format_exc());raise
