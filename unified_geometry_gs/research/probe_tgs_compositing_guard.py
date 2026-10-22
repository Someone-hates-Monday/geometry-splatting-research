"""Retained full images plus explicit native-oracle compositing-boundary tests."""
import hashlib,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from probe_tgs_stable_window import SafeNative,SafeAuxNative,signature
from probe_tgs_adaptive_window import decode,full_reference
from tgs_mixed_numpy import evaluate,complex_gradient
from probe_tgs_radius_contraction import check_group
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compositing_guard_v4';OLD=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def align(n):return (n+127)//128*128

def state(res,f):
    n=f['n'];h=f['h'];w=f['w'];hw=h*w;raw=res[6].cpu().numpy();flags=raw[-128-4*hw:-128].view(np.uint32).reshape(h,w).copy()
    off=align(12*hw);contributors=raw[off:off+8*hw].view(np.uint32).reshape(2,h,w).copy()
    geom=res[4].cpu().numpy();mx=len(geom)-128-8*n;mn=mx-align(8*n);counts=mn-align(12*n);err=counts-align(40*n);coef=err-align(80*n);pre=coef-align(12*n)
    preflight=geom[pre:pre+12*n].view(np.uint32).reshape(n,3).copy()
    assert (flags<=3).all();return flags,contributors,preflight

def modules():
    out={}
    for v in ['adaptive','oracle64']:
        r=json.loads((OUT/v/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256']
        s=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);out[v]=m
    return out

def main():
    assert not (OUT/'results.json').exists() and not (OUT/'progress.json').exists()
    plan=json.loads((OUT/'input_plan_extended.json').read_text());mods=modules();records=[];started=time.time()
    assert plan['protocol_sha256']==sha(ROOT/'TGS_COMPOSITING_GUARD_PROTOCOL_V4.md') and plan['addendum_sha256']==sha(ROOT/'TGS_COMPOSITING_INPUT_ADDENDUM_V4_1.md')
    (OUT/'execution_plan.json').write_text(json.dumps(dict(input_plan_sha256=sha(OUT/'input_plan_extended.json'),script_sha256=sha(Path(__file__)),module_sha256={v:sha(Path(m.__file__)) for v,m in mods.items()},training=False),indent=2)+'\n')
    for ci,e in enumerate(plan['cases']):
        path=ROOT/e['input'];assert sha(path)==e['sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        f['family']=e['family'];new=e['family']=='compositing_cases';patterns=f.get('patterns',np.array([f['pixel_grad']]))
        if not new:
            with np.load(OLD/('reference_'+f['name']+'.npz')) as z:ref={k:z[k].copy() for k in z.files}
        else:
            f['family']='old_cases';rgb,aux=full_reference(f);gs=[];sample=None;fingerprints=[]
            for p in patterns:
                f['pixel_grad']=p;_,sample,fp=evaluate(f['theta'].astype(float),f);gs.append(complex_gradient(f['theta'],f));fingerprints.append(fp)
            ref=dict(rgb=rgb,aux=aux,samples=sample,gradients=np.array(gs));np.savez_compressed(OUT/('cpu_reference_'+f['name']+'.npz'),**ref)
        outputs={};case_rows=[]
        for v,m in mods.items():
            f['pixel_grad']=patterns[0];native=(SafeAuxNative if f['kind']=='aux' else SafeNative)(m,f);data=native.forward(f['theta']);sig=signature(data[1],f);counts,errors,coeff=decode(data[1],f['n']);flags,contributors,pref=state(data[1],f)
            rgb=data[1][1].cpu().numpy();aux=data[1][2].cpu().numpy();radii=data[1][3].cpu().numpy();maxblend=data[1][9].cpu().numpy();grads=[];pr=[]
            for pi,p in enumerate(patterns):
                f['pixel_grad']=p;native.grad.zero_()
                for (x,y),weight in zip(f['pixels'],p):native.grad[:,y,x]=native.t(weight)
                g,_=native.backward(data);grads.append(g);n=f['n'];exact=ref['gradients'][pi]
                groups=dict(geometry=check_group(g[:9*n],exact[:9*n]),sigma_opacity=check_group(g[9*n:11*n],exact[9*n:11*n]))
                if f['degree']>=0:groups['SH']=check_group(g[11*n:],exact[11*n:])
                vg=abs(float(g@f['special_direction']))<=1e-6 if f['check']=='homothety_nullspace' else all(q['gate'] for q in groups.values())
                fe=float(np.abs(data[-1]-ref['samples']).max());pr.append(dict(pattern=pi,CPU_groups=groups,CPU_forward_abs=fe,CPU_gate=bool(vg and fe<=3e-5)))
            full=float(np.abs(rgb-ref['rgb']).max());fullaux=float(np.abs(aux-ref['aux']).max()) if ref['aux'].size else None
            a=dict(native_rgb=rgb,native_aux=aux,native_radii=radii,native_samples=data[-1],native_gradients=np.array(grads),flags=flags,contributors=contributors,preflight=pref,counters=counts,double_coeff=coeff,cache_errors=errors,maxblend=maxblend,rect_min=np.array(sig['rect_min']),rect_max=np.array(sig['rect_max']))
            outputs[v]=a;row=dict(name=f['name'],family=e['family'],variant=v,patterns=pr,CPU_full_rgb_abs=full,CPU_full_aux_abs=fullaux,base_signature=sig,
                flagged_pixels=int((flags>0).sum()),median_flags=int(((flags&1)>0).sum()),early_stop_flags=int(((flags&2)>0).sum()),preflight_totals=pref.astype(np.uint64).sum(0).tolist(),render_totals=counts.astype(np.uint64).sum(0).tolist(),
                CPU_gate=bool(all(p['CPU_gate'] for p in pr) and full<=3e-5),gate=False)
            np.savez_compressed(OUT/(v+'_'+f['name']+'.npz'),**a);case_rows.append(row)
        a,b=outputs['adaptive'],outputs['oracle64'];comparisons=dict(rgb=float(np.abs(a['native_rgb']-b['native_rgb']).max()),aux=float(np.abs(a['native_aux']-b['native_aux']).max()),maxblend=float(np.abs(a['maxblend']-b['maxblend']).max()),radii=bool(np.array_equal(a['native_radii'],b['native_radii'])),contributors=bool(np.array_equal(a['contributors'],b['contributors'])),coefficients=bool(np.array_equal(a['double_coeff'],b['double_coeff'])))
        cmpgrad=[]
        for ga,gb in zip(a['native_gradients'],b['native_gradients']):
            n=f['n'];cmpgrad.append({key:check_group(ga[sl],gb[sl]) for key,sl in [('geometry',slice(0,9*n)),('sigma_opacity',slice(9*n,11*n))]+([('SH',slice(11*n,None))] if f['degree']>=0 else [])})
        ngate=all(all(q['gate'] for q in g.values()) for g in cmpgrad) and comparisons['rgb']<=3e-5 and comparisons['aux']<=3e-5 and comparisons['maxblend']<=3e-5 and comparisons['radii'] and comparisons['contributors'] and comparisons['coefficients']
        for row in case_rows:row.update(native_comparison=comparisons,native_gradient_comparisons=cmpgrad,gate=bool(ngate and (new or row['CPU_gate'])))
        records.extend(case_rows);(OUT/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(dict(case=ci+1,total=len(plan['cases']),name=f['name'],gate=all(r['gate'] for r in case_rows),flags=case_rows[0]['flagged_pixels'],native=comparisons,CPU_gates=[r['CPU_gate'] for r in case_rows])),flush=True)
    final=dict(records=records,cases=len(plan['cases']),vjp_count=sum(len(r['patterns']) for r in records),all_gates=all(r['gate'] for r in records),failed={v:[r['name'] for r in records if r['variant']==v and not r['gate']] for v in mods},input_plan_sha256=sha(OUT/'input_plan_extended.json'),execution_plan_sha256=sha(OUT/'execution_plan.json'),seconds=time.time()-started,training=False,performance_measurement_allowed=False)
    (OUT/'results.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({k:v for k,v in final.items() if k!='records'}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'execution_failure.txt').write_text(traceback.format_exc());raise
