"""Recompute precision and counter-removal gates before any timing is allowed."""
import json
from pathlib import Path
import numpy as np
from tgs_compensated_common import ROOT,OUT,VARIANTS,sha
from probe_tgs_compensated_cases import groups
from tgs_barycentric_reference import evaluate as barycentric
from tgs_mixed_numpy import evaluate as rgb_eval,complex_gradient as rgb_grad
from tgs_aux_numpy import evaluate as aux_eval,complex_gradient as aux_grad
from probe_tgs_radius_contraction import check_group
def main():
    target=OUT/'precision_audit.json';assert not target.exists();plan=json.loads((OUT/'input_plan.json').read_text());cases_result=json.loads((OUT/'case_results.json').read_text());reals=json.loads((OUT/'real_results.json').read_text());fds=json.loads((OUT/'fd_results.json').read_text());binary=json.loads((OUT/'binary_audit.json').read_text())
    assert cases_result['all_gates'] and reals['all_gates'] and fds['all_support_stable'] and binary['all_verified']
    parent=ROOT/'experiments/tgs_compositing_guard_v4/input_plan_extended.json'
    assert plan['parent_plan_sha256']==sha(parent) and plan['cases']==json.loads(parent.read_text())['cases'] and len(plan['cases'])==127
    assert fds['script_sha256']==sha(ROOT/'probe_tgs_compensated_fd.py') and fds['case_results_sha256']==sha(OUT/'case_results.json')
    assert binary['script_sha256']==sha(ROOT/'audit_tgs_compensated_binary.py')
    execution=json.loads((OUT/'case_execution_plan.json').read_text());assert execution['script_sha256']==sha(ROOT/'probe_tgs_compensated_cases.py') and execution['input_plan_sha256']==sha(OUT/'input_plan.json') and cases_result['execution_plan_sha256']==sha(OUT/'case_execution_plan.json')
    assert plan['phase_common_sha256']==sha(ROOT/'tgs_compensated_common.py') and plan['protocol_sha256']==sha(ROOT/'TGS_COMPENSATED_RENDERER_PROTOCOL_V8.md')
    sealed=0
    for name in ['tgs_radius_contraction_v1','tgs_stable_window_v2','tgs_adaptive_window_v3','tgs_compositing_guard_v4','tgs_phase_cost_v5','tgs_compensated_edge_v6','tgs_real_work_v7']:
        for e in json.loads((ROOT/'experiments'/name/'evidence_manifest.json').read_text())['files']:assert sha(ROOT/e['path'])==e['sha256'];sealed+=1
    source=ROOT/'sources/tgs_native_diagnostic_v1';entries=json.loads((source/'manifest.json').read_text())['files']
    for e in entries:assert sha(source/e['path'])==e['sha256']
    for v in VARIANTS:
        r=json.loads((OUT/v/'build_receipt.json').read_text());assert r['generator_sha256']==sha(ROOT/'tgs_compensated_renderer_backend.py') and sha(Path(r['module_file']))==r['module_sha256']==execution['modules'][v]
        for e in r['sources']:assert sha(ROOT/e['path'])==e['sha256']
    helper=(OUT/'generated/native/cuda_rasterizer/stable_window.cuh').read_text();forward=(OUT/'generated/native/cuda_rasterizer/forward.cu').read_text()
    required=['__fsub_rn(pixel.x,anchor.x)','comp_dot2(p+6*k,x,y)','__fsub_rd(r[k],e[k])','__fadd_ru(r[k],e[k])','const float E=e[z.active]','comp_distance_down(z.alpha,1.f/255.f)','__dmul_rn(-best,coeff[9])','__double2float_rn(fmin(.99,raw))']
    assert all(s in helper for s in required) and '__fmul_rn(T,__fsub_rn(1.0f,alpha))' in forward and '#if WINDOW_COUNTERS' in forward
    assert 'stable_error+48*j_id' in forward and 'comp_precompute(stable_error+48*idx+10' in forward
    assert 'obtain(chunk, geom.stable_error, P*48, 128)' in (OUT/'generated/native/cuda_rasterizer/rasterizer_impl.cu').read_text()
    assert 'stable_error+48*window_id' in (OUT/'generated/native/cuda_rasterizer/backward.cu').read_text()
    baseline=json.loads((OUT/'baseline_cache_audit.json').read_text());assert baseline['all_gates'] and baseline['script_sha256']==sha(ROOT/'audit_tgs_compensated_baseline.py')
    index={(r['name'],r['variant']):r for r in cases_result['records']};rows=[];references={};inputs={};pixels=0
    for ci,e in enumerate(plan['cases']):
        path=ROOT/e['input'];assert sha(path)==e['sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        inputs[f['name']]=f;new=e['family']=='compositing_cases';patterns=f.get('patterns',np.array([f['pixel_grad']]));gs=[];ss=[]
        for p in patterns:
            f['pixel_grad']=p;fun,grad=(aux_eval,aux_grad) if f['kind']=='aux' else (rgb_eval,rgb_grad);_,samples,_=fun(f['theta'].astype(float),f);ss.append(samples);gs.append(grad(f['theta'],f))
        references[f['name']]=gs;data={}
        for v in VARIANTS:
            with np.load(OUT/(v+'_'+f['name']+'.npz')) as z:data[v]={k:z[k].copy() for k in z.files}
        maxima={v:0. for v in ['guard','oracle']}
        for start in range(0,f['w']*f['h'],16384):
            ids=np.arange(start,min(start+16384,f['w']*f['h']));pix=np.c_[ids%f['w'],ids//f['w']];rgb=barycentric(f['theta'].astype(float),f,pix)
            for v in maxima:maxima[v]=max(maxima[v],float(np.abs(data[v]['native_rgb'].reshape(3,-1)[:,ids].T-rgb).max()))
        pixels+=f['w']*f['h'];oracle=data['oracle'];parity={}
        for v in ['guard','oracle']:
            aa,bb=data[v],data[v+'_diag'];equal=all(np.array_equal(aa[k],bb[k]) for k in ['native_rgb','native_aux','native_radii','maxblend','flags','contributors','double_coeff','cache_errors']);assert equal
            gg=[groups(a,b,f['n'],f['degree']) for a,b in zip(aa['native_gradients'],bb['native_gradients'])]
            small=all(q['relative']<=1e-5 if q['reference_l2']>1e-10 else q['absolute_max']<=1e-6 for g in gg for q in g.values());assert small
            assert not aa['counters'].any() and not aa['preflight'].any();parity[v]=dict(bitwise=True,VJP_small=True)
        for v,a in data.items():
            rec=index[(f['name'],v)];gg=[];cpu=[]
            for pi,g in enumerate(a['native_gradients']):
                assert np.isfinite(g).all();cg=groups(g,gs[pi],f['n'],f['degree']);ng=groups(g,oracle['native_gradients'][pi],f['n'],f['degree']);gg.append(ng)
                cgate=abs(float(g@f['special_direction']))<=1e-6 if f['check']=='homothety_nullspace' else all(q['gate'] for q in cg.values());cgate=bool(cgate and np.abs(a['native_samples']-ss[pi]).max()<=3e-5);assert cgate==rec['patterns'][pi]['CPU_gate'];cpu.append(cgate)
            native=max(float(np.abs(a[k].astype(float)-oracle[k].astype(float)).max()) for k in ['native_rgb','native_aux','maxblend']);ids_equal=np.array_equal(a['contributors'],oracle['contributors']) and np.array_equal(a['native_radii'],oracle['native_radii']);assert native<=3e-5 and ids_equal
            gate=all(q['gate'] for g in gg for q in g.values()) and (new or all(cpu));assert gate and rec['gate']
            if v in maxima and not new:assert maxima[v]<=3e-5
            rows.append(dict(name=f['name'],variant=v,family=e['family'],native_max_abs=native,independent_RGB_abs=maxima[v.removesuffix('_diag')],CPU_selected_gates=cpu,parity=parity[v.removesuffix('_diag')],gate=True))
        if ci%20==19:print(json.dumps(dict(cases=ci+1,total=len(plan['cases']),pixels=pixels)),flush=True)
    fd_errors=[]
    for d in fds['records']:
        f=inputs[d['name']];pi=d['pattern'];hh=d['step'];vec=f['fd_directions'][pi,d['direction']];xp=(f['theta'].astype(float)+hh*vec).astype(np.float32);xm=(f['theta'].astype(float)-hh*vec).astype(np.float32);actual=(xp.astype(float)-xm.astype(float))/(2*hh)
        assert np.array_equal(actual,d['actual_direction']);expected=float(references[d['name']][pi]@actual);native=float(((np.array(d['plus_samples'])-np.array(d['minus_samples']))*f['patterns'][pi]).sum())/(2*hh)
        assert abs(expected-d['expected'])<1e-8*max(1,abs(expected)) and abs(native-d['native_fd'])<1e-8*max(1,abs(native))
        f['pixel_grad']=f['patterns'][pi];_,_,base=rgb_eval(f['theta'].astype(float),f);cp,_,fp=rgb_eval(xp.astype(float),f);cm,_,fm=rgb_eval(xm.astype(float),f)
        base_sig=index[(d['name'],d['variant'])]['base_signature'];membership=[]
        for sig in [d['plus_signature'],d['minus_signature'],base_sig]:
            tile=f['pixels']//16;mn=np.array(sig['rect_min']);mx=np.array(sig['rect_max']);member=((tile[None]>=mn[:,None])&(tile[None]<mx[:,None])).all(-1).tolist();assert member==sig['selected_tile_members'];membership.append(member)
        smooth=membership[0]==membership[1]==membership[2] and d['plus_signature']['alpha_support']==d['minus_signature']['alpha_support']==base_sig['alpha_support'] and fp==fm==base
        assert smooth==d['smooth'] and smooth;fd_errors.append(abs(native-expected)/max(abs(expected),1e-6))
    real_rows=[];assert reals['plan_sha256']==sha(OUT/'real_plan.json');rplan=json.loads((OUT/'real_plan.json').read_text());assert rplan['script_sha256']==sha(ROOT/'probe_tgs_compensated_real.py')
    assert [c['img_name'] for c in rplan['cameras']]==[c['img_name'] for c in plan['cameras']]
    assert [c['split'] for c in rplan['cameras']]==['train','train','test','test','test','test']
    assert rplan['workload']['checkpoint_sha256']=='9898fce495a105f0aa3fed9f0cbb181a68b1438c5eab046b62c9ab0f55a73620'
    for ci,rr in enumerate(reals['rows']):
        data={}
        for v in VARIANTS:
            with np.load(OUT/f'real_{ci}_{v}.npz') as z:data[v]={k:z[k].copy() for k in z.files}
        for v in ['guard','oracle']:
            assert all(np.array_equal(data[v][k],data[v+'_diag'][k]) for k in ['rgb','aux','radii','maxblend','flags','contributors','double_coeff','cache_errors'])
            assert not data[v]['counters'].any() and not data[v]['preflight'].any()
        a,b=data['guard'],data['oracle'];flags=a['flags']>0;assert np.array_equal(a['contributors'],b['contributors']) and np.array_equal(a['radii'],b['radii'])
        assert all(np.array_equal(a[k][:,flags],b[k][:,flags]) for k in ['rgb','aux','contributors'])
        rgb=float(np.abs(a['rgb']-b['rgb']).max());aux=np.abs(a['aux'].astype(float)-b['aux'].astype(float)).reshape(7,-1).max(1);mb=float(np.abs(a['maxblend']-b['maxblend']).max());assert rgb==rr['RGB_abs'] and aux.tolist()==rr['aux_per_channel_abs'] and mb==rr['maxblend_abs']
        grad={}
        for loss in ['rgb','depth']:
            grad[loss]=[]
            for j in range(6):
                ga=a[f'{loss}_gradient_{j}'].astype(float).ravel();gb=b[f'{loss}_gradient_{j}'].astype(float).ravel();assert np.isfinite(ga).all() and np.isfinite(gb).all();q=check_group(ga,gb);assert q['gate'] and abs(q['relative']-rr['gradient_comparisons'][loss][j]['relative'])<1e-12;grad[loss].append(q)
        assert max(rgb,float(aux.max()),mb)<=3e-5
        assert rr['old_v5_oracle_comparison']['rgb']==0 and max(rr['old_v5_oracle_comparison']['aux'])==0
        real_rows.append(dict(camera=ci,image=rr['image'],split=rr['split'],RGB_abs=rgb,aux_abs=aux.tolist(),maxblend_abs=mb,flagged_pixels=int(flags.sum()),gradient_comparisons=grad,all_counter_parities_bitwise=True,gate=True))
    result=dict(rows=rows,real_rows=real_rows,cases=127,VJPs=cases_result['VJPs'],full_pixels_per_version=pixels,FD_records=len(fd_errors),FD_normalized_quantiles=np.quantile(fd_errors,[0,.5,.9,.99,1]).tolist(),all_gates=True,performance_measurement_allowed=True,prior_sealed_hash_checks=sealed,source_files_checked=len(entries),source_phase_intrinsics_verified=required,binary_audit_sha256=sha(OUT/'binary_audit.json'),baseline_cache_audit_sha256=sha(OUT/'baseline_cache_audit.json'),script_sha256=sha(Path(__file__)),case_results_sha256=sha(OUT/'case_results.json'),real_results_sha256=sha(OUT/'real_results.json'),fd_results_sha256=sha(OUT/'fd_results.json'),training=False,limitations='Numerical oracle/window enclosure assumptions unchanged; single checkpoint; no GT or training quality claim.')
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','real_rows']}),flush=True)
if __name__=='__main__':main()
