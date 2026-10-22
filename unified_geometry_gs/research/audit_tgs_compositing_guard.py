"""Independent full determinant reference, VJP, saved path and real-array audit."""
import hashlib,json
from pathlib import Path
import numpy as np
from tgs_barycentric_reference import evaluate as barycentric
from tgs_mixed_numpy import evaluate as rgb_eval,complex_gradient as rgb_grad
from tgs_aux_numpy import evaluate as aux_eval,complex_gradient as aux_grad
from probe_tgs_radius_contraction import check_group
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compositing_guard_v4';OLD=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def down(v):return float(np.nextafter(np.float32(v),np.float32(-np.inf)))
def up(v):return float(np.nextafter(np.float32(v),np.float32(np.inf)))

def trace_intervals():
    rows=[]
    for ci in range(2):
        base=ROOT/'experiments/tgs_compositing_trace_v1'
        with np.load(base/f'camera_{ci}_adaptive.npz') as z:a=z['trace']
        with np.load(base/f'camera_{ci}_oracle64.npz') as z:b=z['trace']
        lo=hi=1.;caught=[];checked=0
        for i in range(min(np.isfinite(a[:,0]).sum(),np.isfinite(b[:,0]).sum())):
            alpha=float(a[i,5]);reference=float(b[i,5]);err=0. if a[i,7] else 3e-6
            assert abs(alpha-reference)<=err+1e-12,(ci,i,alpha,reference,err)
            if a[i,6] or alpha<1/255:continue
            # Previously independently proven prefix masks coincide before first stop disagreement.
            assert lo<=float(b[i,10])<=hi
            al=max(0.,down(alpha-err));ah=min(float(np.float32(.99)),up(alpha+err))
            nl=down(lo*down(1-ah));nh=up(hi*up(1-al));checked+=1
            assert nl<=float(b[i,11])<=nh
            if nl<float(np.float32(1e-4))<=nh:caught.append(dict(ordinal=i+1,lower=nl,upper=nh,adaptive_test=float(a[i,11]),oracle_test=float(b[i,11])))
            if a[i,13]:break
            lo,hi=nl,nh
        divergence=338 if ci==0 else 760
        assert any(r['ordinal']==divergence for r in caught)
        rows.append(dict(camera=ci,valid_steps=checked,stop_threshold_straddles=caught,original_divergence_caught=True))
    return rows

def main():
    target=OUT/'independent_audit.json';assert not target.exists();plan=json.loads((OUT/'input_plan_extended.json').read_text());results=json.loads((OUT/'results.json').read_text());fds=json.loads((OUT/'fd_results.json').read_text());real=json.loads((OUT/'real_results.json').read_text())
    execution=json.loads((OUT/'execution_plan.json').read_text());assert execution['script_sha256']==sha(ROOT/'probe_tgs_compositing_guard.py') and results['input_plan_sha256']==sha(OUT/'input_plan_extended.json') and results['execution_plan_sha256']==sha(OUT/'execution_plan.json')
    assert plan['original_plan_sha256']==sha(OUT/'input_plan.json') and plan['addendum_sha256']==sha(ROOT/'TGS_COMPOSITING_INPUT_ADDENDUM_V4_1.md')
    assert fds['script_sha256']==sha(ROOT/'probe_tgs_guard_fd.py') and fds['results_sha256']==sha(OUT/'results.json')
    sealed=0
    for name in ['tgs_radius_contraction_v1','tgs_stable_window_v2','tgs_adaptive_window_v3']:
        for e in json.loads((ROOT/'experiments'/name/'evidence_manifest.json').read_text())['files']:
            assert sha(ROOT/e['path'])==e['sha256'];sealed+=1
    sm=ROOT/'sources/tgs_native_diagnostic_v1';sources=json.loads((sm/'manifest.json').read_text())['files']
    for e in sources:assert sha(sm/e['path'])==e['sha256']
    for v in ['adaptive','oracle64']:
        receipt=json.loads((OUT/v/'build_receipt.json').read_text());assert receipt['generator_sha256']==sha(ROOT/'tgs_compositing_guard_backend.py') and receipt['protocol_sha256']==sha(ROOT/'TGS_COMPOSITING_GUARD_PROTOCOL_V4.md')
        assert sha(Path(receipt['module_file']))==receipt['module_sha256']==execution['module_sha256'][v]
        for e in receipt['sources']:assert sha(ROOT/e['path'])==e['sha256']
    index={(r['name'],r['variant']):r for r in results['records']};rows=[];cases={};pixels=0;refs={}
    for ci,e in enumerate(plan['cases']):
        p=ROOT/e['input'];assert sha(p)==e['sha256']
        with np.load(p) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        cases[f['name']]=f;patterns=f.get('patterns',np.array([f['pixel_grad']]));new=e['family']=='compositing_cases'
        rp=OUT/('cpu_reference_'+f['name']+'.npz') if new else OLD/('reference_'+f['name']+'.npz')
        with np.load(rp) as z:ref={k:z[k].copy() for k in z.files}
        samples=[];gradients=[]
        for pi,pattern in enumerate(patterns):
            f['pixel_grad']=pattern;fun,grad=(aux_eval,aux_grad) if f['kind']=='aux' else (rgb_eval,rgb_grad)
            _,s,_=fun(f['theta'].astype(float),f);g=grad(f['theta'],f)
            assert np.linalg.norm(g-ref['gradients'][pi])<=1e-8*max(np.linalg.norm(g),1)+1e-10
            gradients.append(g);samples.append(s)
        refs[f['name']]=gradients;outputs={}
        for v in ['adaptive','oracle64']:
            with np.load(OUT/(v+'_'+f['name']+'.npz')) as z:outputs[v]={k:z[k].copy() for k in z.files}
        maxima={v:0. for v in outputs}
        for start in range(0,f['w']*f['h'],16384):
            ids=np.arange(start,min(start+16384,f['w']*f['h']));pix=np.c_[ids%f['w'],ids//f['w']];rgb=barycentric(f['theta'].astype(float),f,pix)
            for v,a in outputs.items():maxima[v]=max(maxima[v],float(np.abs(a['native_rgb'].reshape(3,-1)[:,ids].T-rgb).max()))
        pixels+=f['w']*f['h'];a,b=outputs['adaptive'],outputs['oracle64'];flags=a['flags']>0
        assert np.array_equal(a['contributors'],b['contributors']) and np.array_equal(a['native_radii'],b['native_radii']) and np.array_equal(a['double_coeff'],b['double_coeff'])
        assert not b['flags'].any() and not b['preflight'].any()
        flagged_equal=all(np.array_equal(a[key][:,flags],b[key][:,flags]) for key in ['native_rgb','native_aux','contributors'])
        assert flagged_equal
        native_deltas={key:float(np.abs(a[key].astype(float)-b[key].astype(float)).max()) for key in ['native_rgb','native_aux','maxblend']}
        assert max(native_deltas.values())<=3e-5
        for v,aa in outputs.items():
            rec=index[(f['name'],v)];assert np.isfinite(aa['native_gradients']).all();passed=[]
            for pi,g in enumerate(gradients):
                n=f['n'];ga=aa['native_gradients'][pi];gb=b['native_gradients'][pi]
                cgroups={key:check_group(ga[sl],g[sl]) for key,sl in [('geometry',slice(0,9*n)),('sigma_opacity',slice(9*n,11*n))]+([('SH',slice(11*n,None))] if f['degree']>=0 else [])}
                ngroups={key:check_group(ga[sl],gb[sl]) for key,sl in [('geometry',slice(0,9*n)),('sigma_opacity',slice(9*n,11*n))]+([('SH',slice(11*n,None))] if f['degree']>=0 else [])}
                cgate=abs(float(ga@f['special_direction']))<=1e-6 if f['check']=='homothety_nullspace' else all(q['gate'] for q in cgroups.values())
                ferr=float(np.abs(aa['native_samples']-samples[pi]).max());cgate=bool(cgate and ferr<=3e-5)
                assert cgate==rec['patterns'][pi]['CPU_gate']
                passed.append(dict(CPU_groups=cgroups,native_groups=ngroups,CPU_forward_abs=ferr,CPU_gate=cgate,native_gate=all(q['gate'] for q in ngroups.values())))
            cfull=float(np.abs(aa['native_rgb']-ref['rgb']).max());assert cfull==rec['CPU_full_rgb_abs']
            gate=all(q['native_gate'] for q in passed) and (new or (all(q['CPU_gate'] for q in passed) and cfull<=3e-5 and maxima[v]<=3e-5))
            assert gate==rec['gate'];rows.append(dict(name=f['name'],variant=v,family=e['family'],patterns=passed,independent_full_RGB_abs=maxima[v],CPU_full_aux_abs=rec['CPU_full_aux_abs'],flagged_bitwise_equal=flagged_equal,flagged_pixels=int(flags.sum()),native_deltas=native_deltas,gate=gate))
        if ci%16==15:print(json.dumps(dict(cases_audited=ci+1,total=len(plan['cases']),pixels=pixels)),flush=True)
    fd_errors=[];fd_failures=[]
    for d in fds['records']:
        f=cases[d['name']];hh=d['step'];pi=d['pattern'];vec=f['fd_directions'][pi,d['direction']];xp=(f['theta'].astype(float)+hh*vec).astype(np.float32);xm=(f['theta'].astype(float)-hh*vec).astype(np.float32);actual=(xp.astype(float)-xm.astype(float))/(2*hh)
        assert np.array_equal(actual,d['actual_direction']);native=float(((np.array(d['plus_samples'])-np.array(d['minus_samples']))*f['patterns'][pi]).sum())/(2*hh);expected=float(refs[d['name']][pi]@actual)
        assert abs(native-d['native_fd'])<1e-8*max(1,abs(native)) and abs(expected-d['expected'])<1e-8*max(1,abs(expected))
        if not d['smooth']:fd_failures.append({k:d[k] for k in ['name','variant','step','pattern','direction']})
        fd_errors.append(abs(native-expected)/max(abs(expected),1e-6))
    realrows=[]
    assert real['plan_sha256']==sha(OUT/'real_plan.json')
    for ci,rr in enumerate(real['rows']):
        arrays={}
        for v in ['adaptive','oracle64']:
            with np.load(OUT/f'real_{ci}_{v}.npz') as z:arrays[v]={k:z[k].copy() for k in z.files}
        a,b=arrays['adaptive'],arrays['oracle64'];flags=a['flags']>0
        assert np.array_equal(a['contributors'],b['contributors']) and np.array_equal(a['radii'],b['radii'])
        assert all(np.array_equal(a[k][:,flags],b[k][:,flags]) for k in ['rgb','aux','contributors'])
        assert int(flags.sum())==rr['metrics']['adaptive']['flagged_pixels']
        channels=np.abs(a['aux'].astype(float)-b['aux'].astype(float)).reshape(7,-1).max(1);rgb=float(np.abs(a['rgb']-b['rgb']).max());mb=float(np.abs(a['maxblend']-b['maxblend']).max())
        assert channels.tolist()==rr['aux_per_channel_max'] and rgb==rr['rgb_max_abs'] and mb==rr['maxblend_max_abs']
        comps={}
        for loss in ['rgb','depth']:
            comps[loss]=[]
            for j in range(6):
                ga=a[f'{loss}_gradient_{j}'].astype(float).ravel();gb=b[f'{loss}_gradient_{j}'].astype(float).ravel();assert np.isfinite(ga).all() and np.isfinite(gb).all()
                comp=check_group(ga,gb);comps[loss].append(comp);assert abs(comp['relative']-rr['gradient_comparisons'][loss][j]['relative'])<1e-12
        x,y=rr['outlier_pixel'];assert a['flags'][y,x]==2 and a['aux'][0,y,x]==b['aux'][0,y,x]
        with np.load(OLD/f'real_{ci}_oracle64.npz') as z:oldrgb=z['rgb'];oldaux=z['aux'];oldradius=z['radii']
        v3diff=dict(rgb=float(np.abs(b['rgb']-oldrgb).max()),aux_per_channel=np.abs(b['aux']-oldaux).reshape(7,-1).max(1).tolist(),radii_identical=bool(np.array_equal(b['radii'],oldradius)))
        totals=a['counters'].astype(np.uint64).sum(0);pt=a['preflight'].astype(np.uint64).sum(0)
        gate=rgb<=3e-5 and channels.max()<=3e-5 and mb<=3e-5 and all(q['gate'] for cc in comps.values() for q in cc)
        assert gate==rr['gate'];realrows.append(dict(camera=ci,RGB_abs=rgb,aux_abs=channels.tolist(),maxblend_abs=mb,flagged_pixels=int(flags.sum()),flagged_fraction=float(flags.mean()),outlier_fixed=True,flagged_bitwise_equal=True,gradient_comparisons=comps,render_counts=totals.tolist(),preflight_counts=pt.tolist(),evaluation_count_ratio=float((pt[0]+totals[0])/totals[0]),v3_oracle_comparison=v3diff,gate=bool(gate)))
    audit=dict(rows=rows,real_rows=realrows,trace_interval_demonstration=trace_intervals(),full_pixels_per_version=pixels,vjp_count=sum(len(r['patterns']) for r in rows),fd_count=len(fd_errors),fd_support_failures=fd_failures,fd_normalized_quantiles=np.quantile(fd_errors,[0,.5,.9,.99,1]).tolist(),all_precision_gates=all(r['gate'] for r in rows) and all(r['gate'] for r in realrows) and not fd_failures,previous_sealed_hash_checks=sealed,source_files_checked=len(sources),script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'results.json'),fd_results_sha256=sha(OUT/'fd_results.json'),real_results_sha256=sha(OUT/'real_results.json'),training=False,performance_measurement_allowed=False,
        limitations='Conditional alpha/power enclosure, numerical double geometry, heuristic tile guard. Two-pass work is measured by counters, not claimed accelerated. Native-oracle target for32 new threshold cases; all CPU disagreements retained.')
    target.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
