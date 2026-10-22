"""All 95 fixed inputs, full images, selective counts and retained VJPs."""
import hashlib,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_mixed_numpy import evaluate as rgb_eval,complex_gradient as rgb_grad
from tgs_aux_numpy import evaluate as aux_eval,complex_gradient as aux_grad
from probe_tgs_stable_window import SafeNative,SafeAuxNative,rectangles,signature
from probe_tgs_radius_contraction import check_group
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def decode(res,n):
    raw=res[4].cpu().numpy();mx=len(raw)-128-8*n;mn=mx-((8*n+127)//128)*128
    countoff=mn-((12*n+127)//128)*128;erroroff=countoff-((40*n+127)//128)*128;coeffoff=erroroff-((80*n+127)//128)*128
    counters=raw[countoff:countoff+12*n].view(np.uint32).reshape(n,3).copy()
    errors=raw[erroroff:erroroff+40*n].view(np.float32).reshape(n,10).copy()
    coeffs=raw[coeffoff:coeffoff+80*n].view(np.float64).reshape(n,10).copy()
    visible=res[3].cpu().numpy()>0;errors[~visible]=0;coeffs[~visible]=0
    assert (counters[:,1:]<=counters[:,:1]).all()
    return counters,errors,coeffs

def full_reference(f):
    rgb=np.empty((3,f['h'],f['w']),float);aux=np.empty((7,f['h'],f['w']),float) if f['family']=='old_cases' else None
    for start in range(0,f['h']*f['w'],16384):
        ids=np.arange(start,min(start+16384,f['h']*f['w']));pixels=np.c_[ids%f['w'],ids//f['w']]
        ff=dict(f,pixels=pixels,pixel_grad=np.zeros((len(ids),3)))
        _,samples,_=rgb_eval(f['theta'].astype(float),ff);rgb.reshape(3,-1)[:,ids]=samples.T
        if aux is not None:
            ff['aux_grad']=np.zeros((len(ids),7));_,a,_=aux_eval(f['theta'].astype(float),ff);aux.reshape(7,-1)[:,ids]=a.T
    return rgb,aux

def main():
    assert not (OUT/'results.json').exists() and not (OUT/'progress.json').exists()
    plan=json.loads((OUT/'input_plan.json').read_text());mods={}
    assert plan['protocol_sha256']==sha(ROOT/'TGS_ADAPTIVE_WINDOW_PROTOCOL_V3.md')
    for variant in ['arithmetic64','adaptive','oracle64']:
        r=json.loads((OUT/variant/'build_receipt.json').read_text());p=Path(r['module_file']);assert sha(p)==r['module_sha256']
        spec=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[variant]=m
    execution=dict(script_sha256=sha(Path(__file__)),input_plan_sha256=sha(OUT/'input_plan.json'),modules={k:sha(Path(m.__file__)) for k,m in mods.items()},training=False)
    (OUT/'execution_plan.json').write_text(json.dumps(execution,indent=2)+'\n')
    records=[];started=time.time()
    for ci,e in enumerate(plan['cases']):
        p=ROOT/e['input'];assert sha(p)==e['sha256']
        with np.load(p) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        f['family']=e['family'];fun,grad=(aux_eval,aux_grad) if f['kind']=='aux' else (rgb_eval,rgb_grad)
        patterns=f.get('patterns',np.array([f['pixel_grad']]));refs=[]
        for pattern in patterns:
            f['pixel_grad']=pattern;_,samples,fp=fun(f['theta'].astype(float),f);refs.append((samples,grad(f['theta'],f),fp))
        ideal,ideal_aux=full_reference(f)
        np.savez_compressed(OUT/('reference_'+f['name']+'.npz'),rgb=ideal,aux=ideal_aux if ideal_aux is not None else np.array([]),samples=refs[0][0],gradients=np.array([r[1] for r in refs]))
        images={}
        for variant,mod in mods.items():
            f['pixel_grad']=patterns[0];native=(SafeAuxNative if f['kind']=='aux' else SafeNative)(mod,f);data=native.forward(f['theta'])
            sig=signature(data[1],f);counts,errors,coeff=decode(data[1],f['n']);n=f['n'];rows=[];grads=[]
            for pi,pattern in enumerate(patterns):
                f['pixel_grad']=pattern;native.grad.zero_()
                for (x,y),g in zip(f['pixels'],pattern):native.grad[:,y,x]=native.t(g)
                g,_=native.backward(data);grads.append(g);exact=refs[pi][1]
                groups=dict(geometry=check_group(g[:9*n],exact[:9*n]),sigma_opacity=check_group(g[9*n:11*n],exact[9*n:11*n]))
                if f['degree']>=0:groups['SH']=check_group(g[11*n:],exact[11*n:])
                special=float(g@f['special_direction']) if f['check']=='homothety_nullspace' else None
                vg=abs(special)<=1e-6 if special is not None else all(v['gate'] for v in groups.values())
                fe=float(np.abs(data[-1]-refs[pi][0]).max())
                rows.append(dict(pattern=pi,groups=groups,null_direction=special,forward_abs=fe,gate=bool(vg and fe<=3e-5 and (data[1][3]>0).all())))
            rgb=data[1][1].cpu().numpy();aux=data[1][2].cpu().numpy();rad=data[1][3].cpu().numpy();images[variant]=rgb
            err=np.abs(rgb-ideal);ae=np.abs(aux-ideal_aux) if ideal_aux is not None else None
            maxidx=np.unravel_index(err.argmax(),err.shape)
            row=dict(name=f['name'],family=e['family'],variant=variant,patterns=rows,base_signature=sig,
                full_rgb_abs=float(err.max()),full_rgb_mae=float(err.mean()),full_rgb_bad_pixels=int((err.max(0)>3e-5).sum()),worst_pixel=[int(maxidx[2]),int(maxidx[1])],
                full_aux_abs=float(ae.max()) if ae is not None else None,counter_totals=counts.astype(np.uint64).sum(0).tolist(),
                gate=bool(all(r['gate'] for r in rows) and err.max()<=3e-5))
            records.append(row)
            np.savez_compressed(OUT/(variant+'_'+f['name']+'.npz'),native_rgb=rgb,native_aux=aux,native_radii=rad,native_samples=data[-1],native_gradients=np.array(grads),counters=counts,cache_errors=errors,double_coeff=coeff,rect_min=np.array(sig['rect_min']),rect_max=np.array(sig['rect_max']))
        delta=float(np.abs(images['adaptive']-images['oracle64']).max())
        for row in records[-3:]:row['adaptive_oracle_rgb_max_delta']=delta
        (OUT/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(dict(case=ci+1,total=len(plan['cases']),name=f['name'],outcomes={r['variant']:dict(gate=r['gate'],full_max=r['full_rgb_abs'],max_geom=max(p['groups']['geometry']['relative'] for p in r['patterns']),counts=r['counter_totals']) for r in records[-3:]})),flush=True)
    final=dict(cases=len(plan['cases']),records=records,vjp_count=sum(len(r['patterns']) for r in records),failed={v:[r['name'] for r in records if r['variant']==v and not r['gate']] for v in mods},candidate_gates_passed=all(r['gate'] for r in records if r['variant'] in ['adaptive','oracle64']),input_plan_sha256=sha(OUT/'input_plan.json'),execution_plan_sha256=sha(OUT/'execution_plan.json'),seconds=time.time()-started,performance_measurement_allowed=False,training=False)
    (OUT/'results.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({k:v for k,v in final.items() if k!='records'}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'execution_failure.txt').write_text(traceback.format_exc());raise
