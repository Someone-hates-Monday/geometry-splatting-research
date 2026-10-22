"""Independently recompute each candidate's full precision decision; retain failures."""
import hashlib,json,traceback
from pathlib import Path
import numpy as np
from tgs_window_candidates_common import ROOT,OUT,BASES,VARIANTS,sha,strict_groups
from probe_tgs_window_candidates_cases import groups
from tgs_barycentric_reference import evaluate as barycentric
from tgs_mixed_numpy import evaluate as rgb_eval,complex_gradient as rgb_grad
from tgs_aux_numpy import evaluate as aux_eval,complex_gradient as aux_grad
from probe_tgs_radius_contraction import check_group
def main():
    target=OUT/'precision_audit.json';assert not target.exists();plan=json.loads((OUT/'input_plan.json').read_text());cases=json.loads((OUT/'case_results.json').read_text());real=json.loads((OUT/'real_results.json').read_text());fd=json.loads((OUT/'fd_results.json').read_text());cache=json.loads((OUT/'cache_audit.json').read_text());binary=json.loads((OUT/'binary_audit.json').read_text());assert binary['all_verified'];assert plan['protocol_sha256']==sha(ROOT/'TGS_WINDOW_CANDIDATES_PROTOCOL_V9.md');assert plan['common_sha256']==sha(ROOT/'tgs_window_candidates_common.py');assert plan['cases']==json.loads((ROOT/'experiments/tgs_compensated_renderer_v8/input_plan.json').read_text())['cases']
    for path,h in plan['sources'].items():assert sha(ROOT/path)==h
    for path,h in plan['test_scripts'].items():assert sha(ROOT/path)==h
    prior=0
    for directory,expected in plan['prior_manifests'].items():
        p=ROOT/'experiments'/directory/'evidence_manifest.json';assert sha(p)==expected
        for en in json.loads(p.read_text())['files']:assert sha(ROOT/en['path'])==en['sha256'];prior+=1
    source=ROOT/'sources/tgs_native_diagnostic_v1';original=json.loads((source/'manifest.json').read_text())['files']
    for en in original:assert sha(source/en['path'])==en['sha256']
    for v in VARIANTS:
        folder=ROOT/'experiments/tgs_phase_cost_v5' if v.startswith('oracle') else OUT;receipt=json.loads((folder/v/'build_receipt.json').read_text());assert sha(Path(receipt['module_file']))==receipt['module_sha256']
    cw=(OUT/'coarse/generated/native/cuda_rasterizer/stable_window.cuh').read_text();cf=(OUT/'coarse/generated/native/cuda_rasterizer/forward.cu').read_text();sw=(OUT/'standard/generated/native/cuda_rasterizer/stable_window.cuh').read_text();assert 'if(false) { // coarse: no preflight' in cf and 'coarse_precompute(stable_error+14*idx' in cf and 'stable_error+14*j_id' in cf and '__dmul_rn(-best,c[9])' in cw
    assert '__nv_powf(fmaxf(z.u,0.f),sigma)' in sw and '__powf(' not in sw
    base8=(ROOT/'tgs_compensated_window_v8.cuh').read_text();assert sw==base8.replace('struct StableWindow','extern "C" __device__ float __nv_powf(float,float);\nstruct StableWindow').replace('__powf(fmaxf(z.u,0.f),sigma)','__nv_powf(fmaxf(z.u,0.f),sigma)')
    idx={(r['name'],r['variant']):r for r in cases['records']};refs={};inputs={};rows=[];pixels=0;candidate_case={v:True for v in BASES}
    for ci,en in enumerate(plan['cases']):
        inp=ROOT/en['input'];assert sha(inp)==en['sha256']
        with np.load(inp) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        inputs[f['name']]=f;patterns=f.get('patterns',np.array([f['pixel_grad']]));gs=[];ss=[]
        for p in patterns:
            f['pixel_grad']=p;fun,gr=(aux_eval,aux_grad) if f['kind']=='aux' else (rgb_eval,rgb_grad);_,samples,_=fun(f['theta'].astype(float),f);ss.append(samples);gs.append(gr(f['theta'],f))
        refs[f['name']]=gs;data={}
        for v in VARIANTS:
            with np.load(OUT/(v+'_'+f['name']+'.npz')) as z:data[v]={k:z[k] for k in z.files}
        b=data['oracle'];maximum={v:0. for v in BASES}
        for start in range(0,f['w']*f['h'],16384):
            ids=np.arange(start,min(start+16384,f['w']*f['h']));pix=np.c_[ids%f['w'],ids//f['w']];physical=barycentric(f['theta'].astype(float),f,pix)
            for v in BASES:maximum[v]=max(maximum[v],float(np.abs(data[v]['native_rgb'].reshape(3,-1)[:,ids].T-physical).max()))
        pixels+=f['w']*f['h'];parity={}
        for v in BASES:
            a,dd=data[v],data[v+'_diag'];equal=all(np.array_equal(a[k],dd[k]) for k in ['native_rgb','native_aux','native_radii','maxblend','flags','contributors','double_coeff','cache_errors']);pg=[groups(x,y,f['n'],f['degree']) for x,y in zip(a['native_gradients'],dd['native_gradients'])];strict=all(strict_groups(list(g.values())) for g in pg);assert equal and strict;parity[v]=True
        for v,a in data.items():
            base=v.removesuffix('_diag');record=idx[(f['name'],v)];ng=[];cpu=[]
            for pi,g in enumerate(a['native_gradients']):
                assert np.isfinite(g).all();cg=groups(g,gs[pi],f['n'],f['degree']);gg=groups(g,b['native_gradients'][pi],f['n'],f['degree']);ng.append(gg);cgok=abs(float(g@f['special_direction']))<=1e-6 if f['check']=='homothety_nullspace' else all(q['gate'] for q in cg.values());cpu.append(bool(cgok and np.abs(a['native_samples']-ss[pi]).max()<=3e-5));assert cpu[-1]==record['patterns'][pi]['CPU_gate']
            delta=max(float(np.abs(a[k].astype(float)-b[k].astype(float)).max()) for k in ['native_rgb','native_aux','maxblend']);same=np.array_equal(a['contributors'],b['contributors']) and np.array_equal(a['native_radii'],b['native_radii']);gate=delta<=3e-5 and same and all(q['gate'] for g in ng for q in g.values()) and (en['family']=='compositing_cases' or all(cpu))
            if base=='coarse':gate &= delta==0 and all(strict_groups(list(g.values())) for g in ng) and not a['flags'].any() and not a['preflight'].any()
            assert bool(gate)==bool(record['gate']);candidate_case[base]&=bool(gate);rows.append(dict(name=f['name'],variant=v,native_max_abs=delta,independent_RGB_abs=maximum[base],gate=bool(gate)))
        if ci%20==19:print(json.dumps(dict(cases=ci+1,pixels=pixels)),flush=True)
    assert candidate_case==cases['candidate_gates'];fde={v:[] for v in BASES};fdsupport={v:True for v in BASES}
    for en in fd['records']:
        f=inputs[en['name']];pi=en['pattern'];hh=en['step'];vec=f['fd_directions'][pi,en['direction']];xp=(f['theta'].astype(float)+hh*vec).astype(np.float32);xm=(f['theta'].astype(float)-hh*vec).astype(np.float32);actual=(xp.astype(float)-xm.astype(float))/(2*hh);assert np.array_equal(actual,en['actual_direction']);expected=float(refs[en['name']][pi]@actual);value=float(((np.array(en['plus_samples'])-np.array(en['minus_samples']))*f['patterns'][pi]).sum())/(2*hh);assert abs(expected-en['expected'])<1e-8*max(1,abs(expected)) and abs(value-en['native_fd'])<1e-8*max(1,abs(value))
        f['pixel_grad']=f['patterns'][pi];_,_,basefp=rgb_eval(f['theta'].astype(float),f);_,_,fp=rgb_eval(xp.astype(float),f);_,_,fm=rgb_eval(xm.astype(float),f);base_sig=idx[(en['name'],en['variant'])]['base_signature'];members=[]
        for sig in [en['plus_signature'],en['minus_signature'],base_sig]:
            tile=f['pixels']//16;mn=np.array(sig['rect_min']);mx=np.array(sig['rect_max']);member=((tile[None]>=mn[:,None])&(tile[None]<mx[:,None])).all(-1).tolist();assert member==sig['selected_tile_members'];members.append(member)
        stable=members[0]==members[1]==members[2] and en['plus_signature']['alpha_support']==en['minus_signature']['alpha_support']==base_sig['alpha_support'] and fp==fm==basefp;assert stable==en['smooth'];fdsupport[en['variant']]&=stable;fde[en['variant']].append(abs(value-expected)/max(abs(expected),1e-6))
    rrows=[];rgates={v:True for v in ['coarse','standard']}
    assert len(real['rows'])==7 and real['model_inputs_unchanged'];assert real['rows'][-1]['name']==plan['cameras'][-1]['img_name'] and real['rows'][-1]['new_holdout']
    for ci,rr in enumerate(real['rows']):
        with np.load(OUT/f'real_{ci}_oracle.npz') as z:b={k:z[k] for k in z.files}
        metrics={}
        for v in ['coarse','standard','oracle']:
            with np.load(OUT/f'real_{ci}_{v}.npz') as z:a={k:z[k] for k in z.files}
            with np.load(OUT/f'real_{ci}_{v}_diag.npz') as z:
                assert all(np.array_equal(a[k],z[k]) for k in ['rgb','aux','radii','maxblend','flags','contributors','double_coeff','cache_errors']);diag_preflight_zero=not z['preflight'].any()
            assert not a['counters'].any() and not a['preflight'].any()
            if v=='oracle':continue
            q=rr['candidates'][v];rgb=float(np.abs(a['rgb']-b['rgb']).max());aux=np.abs(a['aux'].astype(float)-b['aux'].astype(float)).reshape(7,-1).max(1);mb=float(np.abs(a['maxblend']-b['maxblend']).max());assert rgb==q['RGB_abs'] and aux.tolist()==q['aux_per_channel_abs'] and mb==q['maxblend_abs'];paths=np.array_equal(a['contributors'],b['contributors']) and np.array_equal(a['radii'],b['radii']);flags=a['flags']>0;flagged=all(np.array_equal(a[k][:,flags],b[k][:,flags]) for k in ['rgb','aux','contributors']);gg={}
            for loss in ['rgb','depth']:
                gg[loss]=[]
                for j in range(6):
                    x,y=a[f'{loss}_gradient_{j}'].astype(float).ravel(),b[f'{loss}_gradient_{j}'].astype(float).ravel();assert np.isfinite(x).all() and np.isfinite(y).all();g=check_group(x,y);assert abs(g['relative']-q['gradient_groups'][loss][j]['relative'])<1e-12;gg[loss].append(g)
            gate=max(rgb,float(aux.max()),mb)<=3e-5 and paths and flagged and all(g['gate'] for gs in gg.values() for g in gs)
            if v=='coarse':gate &= max(rgb,float(aux.max()),mb)==0 and not flags.any() and diag_preflight_zero and all(strict_groups(gs) for gs in gg.values())
            oldbad=json.loads((ROOT/'experiments/tgs_compensated_renderer_v8/depth_failure_plan.json').read_text())['rows']
            expected_bad=next((r['bad'] for r in oldbad if r['name']==rr['name']),[]);assert len(q['regression_14_pixels'])==len(expected_bad)
            for oldpt,newpt in zip(expected_bad,q['regression_14_pixels']):
                assert oldpt['pixel']==newpt['pixel'];xx,yy=oldpt['pixel'];assert abs(float(a['aux'][0,yy,xx])-float(b['aux'][0,yy,xx]))==newpt['depth_abs']
            assert bool(gate)==bool(q['gate']);rgates[v]&=bool(gate);metrics[v]=dict(RGB_abs=rgb,aux_abs=aux.tolist(),maxblend_abs=mb,gradient_groups=gg,gate=bool(gate))
        rrows.append(dict(camera=ci,name=rr['name'],candidates=metrics));print(json.dumps(dict(real_camera=ci,gates={v:m['gate'] for v,m in metrics.items()})),flush=True)
    assert rgates==real['candidate_gates'];decisions={v:bool(candidate_case[v] and rgates[v] and fdsupport[v] and fdsupport['oracle'] and cache['candidate_gates'][v]) for v in ['coarse','standard']}
    output=dict(cases=127,VJPs=cases['VJPs'],pixels_per_version=pixels,rows=rows,real_rows=rrows,FD_records_by_variant={v:len(x) for v,x in fde.items()},FD_support_stable=fdsupport,FD_normalized_quantiles={v:np.quantile(x,[0,.5,.9,.99,1]).tolist() for v,x in fde.items()},candidate_gates=decisions,all_gates=all(decisions.values()),performance_measurement_allowed=decisions,prior_hash_checks=prior,original_source_hash_checks=len(original),script_sha256=sha(Path(__file__)),case_results_sha256=sha(OUT/'case_results.json'),real_results_sha256=sha(OUT/'real_results.json'),fd_results_sha256=sha(OUT/'fd_results.json'),cache_audit_sha256=sha(OUT/'cache_audit.json'),binary_audit_sha256=sha(OUT/'binary_audit.json'),training=False,power_interval_proved_for_standard=False)
    target.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items() if k not in ['rows','real_rows']}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'precision_audit_failure.txt').write_text(traceback.format_exc());raise
