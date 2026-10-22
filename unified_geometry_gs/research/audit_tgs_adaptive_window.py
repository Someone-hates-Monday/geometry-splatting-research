"""Independent determinant geometry, Torch VJP, exact cache-bound and source audit."""
from fractions import Fraction as F
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import sympy as sp
import torch
from tgs_barycentric_reference import evaluate as independent_rgb
from tgs_mixed_reference import reference as rgb_reference
from tgs_aux_reference import reference as aux_reference
from tgs_mixed_numpy import evaluate as fd_reference
from probe_tgs_stable_window import SafeNative,rectangles
from probe_tgs_adaptive_window import decode
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def align(x):return (x+127)//128*128

def cache_audit(mod,f,saved):
    data=SafeNative(mod,f).forward(f['theta']);res=data[1];n=f['n'];counts,err,coeff=decode(res,n)
    assert np.array_equal(counts,saved['counters']) and np.array_equal(err,saved['cache_errors']) and np.array_equal(coeff,saved['double_coeff'])
    raw=res[4].cpu().numpy();mx=len(raw)-128-8*n;mn=mx-align(8*n);co=mn-align(12*n);eo=co-align(40*n);so=eo-align(80*n)
    pw=so-align(12*n);no=pw-align(24*n);off=no-align(12*n)
    norms=raw[no:no+24*n].view(np.float32).reshape(n,6);offsets=raw[off:off+12*n].view(np.float32).reshape(n,3)
    pos=0
    for size in [4*n,3*n,4*n,8*n,16*n]:pos=align(pos)+size
    pc=align(pos);rinv=-raw[pc:pc+8*n].view(np.float32).reshape(n,2)[:,0]
    floats=np.c_[norms,offsets,rinv]
    visible=res[3].cpu().numpy()>0;checked=0
    for i in np.flatnonzero(visible):
        for j in range(10):
            exact_error=abs(F(float(coeff[i,j]))-F(float(floats[i,j])))
            assert F(float(err[i,j]))>=exact_error,(f['name'],i,j,float(err[i,j]),float(exact_error))
            checked+=1
    mn,mx=rectangles(res,n,f['w'],f['h'])
    np.savez_compressed(OUT/('bound_audit_'+str(f['name'])+'.npz'),float_coeff=floats,double_coeff=coeff,error_bound=err,counts=counts,visible=visible,rect_min=mn,rect_max=mx)
    return checked

def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    plan=json.loads((OUT/'input_plan.json').read_text());results=json.loads((OUT/'results.json').read_text());fds=json.loads((OUT/'fd_results.json').read_text())
    execution=json.loads((OUT/'execution_plan.json').read_text())
    assert results['input_plan_sha256']==sha(OUT/'input_plan.json') and results['execution_plan_sha256']==sha(OUT/'execution_plan.json')
    assert execution['script_sha256']==sha(ROOT/'probe_tgs_adaptive_window.py') and plan['protocol_sha256']==sha(ROOT/'TGS_ADAPTIVE_WINDOW_PROTOCOL_V3.md')
    assert fds['script_sha256']==sha(ROOT/'probe_tgs_adaptive_fd.py') and fds['results_sha256']==sha(OUT/'results.json')
    manifests=['experiments/tgs_radius_contraction_v1/evidence_manifest.json','experiments/tgs_stable_window_v2/evidence_manifest.json']
    previous_count=0
    for name in manifests:
        for e in json.loads((ROOT/name).read_text())['files']:
            assert sha(ROOT/e['path'])==e['sha256'];previous_count+=1
    sm=ROOT/'sources/tgs_native_diagnostic_v1';source_entries=json.loads((sm/'manifest.json').read_text())['files']
    for e in source_entries:assert sha(sm/e['path'])==e['sha256']
    mod=None
    for v in ['arithmetic64','adaptive','oracle64']:
        r=json.loads((OUT/v/'build_receipt.json').read_text());assert r['generator_sha256']==sha(ROOT/'tgs_adaptive_window_backend.py') and r['helper_sha256']==sha(ROOT/'tgs_adaptive_window.cuh')
        p=Path(r['module_file']);assert sha(p)==r['module_sha256']==execution['modules'][v]
        for e in r['sources']:assert sha(ROOT/e['path'])==e['sha256']
        if v=='adaptive':
            spec=importlib.util.spec_from_file_location(p.stem,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    index={(r['name'],r['variant']):r for r in results['records']};rows=[];pixels_total=0;cache_checks=0;fd_errors=[];fd_bad=[]
    theta_refs={};case_inputs={}
    for ci,e in enumerate(plan['cases']):
        p=ROOT/e['input'];assert sha(p)==e['sha256']
        with np.load(p) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        case_inputs[f['name']]=f
        patterns=f.get('patterns',np.array([f['pixel_grad']]))
        with np.load(OUT/('reference_'+f['name']+'.npz')) as z:cached={k:z[k].copy() for k in z.files}
        refs=[]
        for pi,pattern in enumerate(patterns):
            f['pixel_grad']=pattern;x=torch.tensor(f['theta'].astype(float),requires_grad=True)
            loss,samples,meta=(aux_reference if f['kind']=='aux' else rgb_reference)(x,f);g=torch.autograd.grad(loss,x)[0].numpy()
            assert np.linalg.norm(g-cached['gradients'][pi])<1e-6*max(np.linalg.norm(g),1e-10)+1e-10
            refs.append(g)
        theta_refs[f['name']]=refs
        outputs={}
        for v in ['arithmetic64','adaptive','oracle64']:
            with np.load(OUT/(v+'_'+f['name']+'.npz')) as z:outputs[v]={k:z[k].copy() for k in z.files}
        full_error={v:0. for v in outputs};reference_delta=0.;badpixels={v:0 for v in outputs}
        for start in range(0,f['w']*f['h'],16384):
            ids=np.arange(start,min(start+16384,f['w']*f['h']));pix=np.c_[ids%f['w'],ids//f['w']]
            rgb=independent_rgb(f['theta'].astype(float),f,pix)
            reference_delta=max(reference_delta,float(np.abs(cached['rgb'].reshape(3,-1)[:,ids].T-rgb).max()))
            for v,a in outputs.items():
                err=np.abs(a['native_rgb'].reshape(3,-1)[:,ids].T-rgb)
                full_error[v]=max(full_error[v],float(err.max()));badpixels[v]+=int((err.max(1)>3e-5).sum())
        pixels_total+=f['w']*f['h'];cache_checks+=cache_audit(mod,f,outputs['adaptive'])
        for v,a in outputs.items():
            assert np.isfinite(a['native_gradients']).all() and np.isfinite(a['native_rgb']).all() and np.isfinite(a['native_aux']).all()
            rec=index[(f['name'],v)];groups=[]
            for pi,g in enumerate(refs):
                n=f['n'];actual=a['native_gradients'][pi];gg=[]
                for sl in [slice(0,9*n),slice(9*n,11*n)]+([slice(11*n,None)] if f['degree']>=0 else []):
                    norm=float(np.linalg.norm(g[sl]));rel=float(np.linalg.norm(actual[sl]-g[sl])/max(norm,1e-12));ab=float(np.abs(actual[sl]-g[sl]).max())
                    gg.append(dict(relative=rel,absolute=ab,gate=bool(rel<=1e-3 if norm>1e-10 else ab<=1e-6)))
                if f['check']=='homothety_nullspace':gate=abs(float(actual@f['special_direction']))<=1e-6
                else:gate=all(q['gate'] for q in gg)
                fe=float(np.abs(a['native_samples']-cached['samples']).max());gate=gate and fe<=3e-5 and bool((a['native_radii']>0).all())
                assert gate==rec['patterns'][pi]['gate'];groups.append(dict(groups=gg,forward_abs=fe,gate=bool(gate)))
            # Recompute the declared original-reference whole-image gate exactly.
            original_error=float(np.abs(a['native_rgb']-cached['rgb']).max());assert original_error==rec['full_rgb_abs']
            declared=all(q['gate'] for q in groups) and original_error<=3e-5;assert declared==rec['gate']
            rows.append(dict(name=f['name'],variant=v,family=e['family'],patterns=groups,independent_full_rgb_abs=full_error[v],independent_bad_pixels=badpixels[v],original_reference_difference=reference_delta,
                full_aux_abs=rec['full_aux_abs'],counters=a['counters'].astype(np.uint64).sum(0).tolist(),gate=bool(declared and full_error[v]<=3e-5)))
        if ci%12==11:print(json.dumps(dict(audited=ci+1,total=len(plan['cases']),pixels=pixels_total)),flush=True)
    for d in fds['records']:
        f=case_inputs[d['name']];pi=d['pattern'];hh=d['step'];vec=f['fd_directions'][pi,d['direction']]
        xp=(f['theta'].astype(float)+hh*vec).astype(np.float32);xm=(f['theta'].astype(float)-hh*vec).astype(np.float32);actual=(xp.astype(float)-xm.astype(float))/(2*hh)
        assert np.array_equal(actual,d['actual_direction'])
        pattern=f['patterns'][pi];f['pixel_grad']=pattern;expected=float(theta_refs[d['name']][pi]@actual)
        native=float(((np.array(d['plus_samples'])-np.array(d['minus_samples']))*pattern).sum())/(2*hh)
        assert abs(native-d['native_fd'])<1e-8*max(1,abs(native))
        assert abs(expected-d['expected'])<1e-6*max(1,abs(expected))
        cp,_,fp=fd_reference(xp.astype(float),f);cm,_,fm=fd_reference(xm.astype(float),f);_,_,base=fd_reference(f['theta'].astype(float),f)
        rec=index[(d['name'],d['variant'])];base_sig=rec['base_signature'];members=[]
        for sig in [d['plus_signature'],d['minus_signature'],base_sig]:
            mn=np.array(sig['rect_min']);mx=np.array(sig['rect_max']);tile=f['pixels']//16
            membership=((tile[None]>=mn[:,None])&(tile[None]<mx[:,None])).all(-1).tolist();assert membership==sig['selected_tile_members'];members.append(membership)
        tile=members[0]==members[1]==members[2];alpha=d['plus_signature']['alpha_support']==d['minus_signature']['alpha_support']==base_sig['alpha_support'];kernel=fp==fm==base
        smooth=tile and alpha and kernel;assert smooth==d['smooth']
        if not smooth:fd_bad.append(dict(name=d['name'],variant=d['variant'],pattern=pi,step=hh,direction=d['direction']))
        fd_errors.append(abs(native-expected)/max(abs(expected),1e-6))
    s,t,o,c=sp.symbols('s t o c',positive=True);uncapped=o*sp.exp(s*t)
    assert sp.diff(uncapped,t)==s*uncapped
    math=dict(log_window_lipschitz='For g(t)=min(c,o*exp(s*t)), derivative is 0 or s*g(t)<=s*c; continuity at cap gives global Lipschitz c*s.',
        relative_u_bound='For |u-uhat|<=d<uhat, |log(u)-log(uhat)|<=d/(uhat-d).',
        cache_error_bounds_exact_rational_checks=cache_checks,limitations='Stored double coefficients are numerical, pow allowance empirical, tile outward guard not universally proved. No full real-scene/training/quality or speed claim.')
    final=dict(rows=rows,math=math,all_candidate_gates=all(r['gate'] for r in rows if r['variant'] in ['adaptive','oracle64']),failed={v:[r['name'] for r in rows if r['variant']==v and not r['gate']] for v in ['arithmetic64','adaptive','oracle64']},full_rgb_pixels_per_version=pixels_total,vjp_count=sum(len(r['patterns']) for r in rows),fd_count=len(fd_errors),fd_support_failures=fd_bad,fd_normalized_quantiles=np.quantile(fd_errors,[0,.5,.9,.99,1]).tolist(),
        previous_sealed_hash_checks=previous_count,source_files_checked=len(source_entries),script_sha256=sha(Path(__file__)),independent_reference_sha256=sha(ROOT/'tgs_barycentric_reference.py'),results_sha256=sha(OUT/'results.json'),fd_results_sha256=sha(OUT/'fd_results.json'),training=False,performance_measurement_allowed=False)
    target.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps({k:v for k,v in final.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
