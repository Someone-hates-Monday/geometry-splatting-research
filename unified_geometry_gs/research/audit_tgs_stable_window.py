"""Independent NumPy re-evaluation, saved-array audit and exact support lemmas."""
import hashlib
import json
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import sympy as sp
from tgs_mixed_numpy import evaluate as rgb_eval,complex_gradient as rgb_grad
from tgs_aux_numpy import evaluate as aux_eval,complex_gradient as aux_grad

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_stable_window_v2'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def check_group(g,r):
    nr=float(np.linalg.norm(r));err=float(np.linalg.norm(g-r)/max(nr,1e-12));a=float(np.abs(g-r).max())
    return dict(relative=err,absolute_max=a,gate=bool(err<=1e-3 if nr>1e-10 else a<=1e-6))


def members(sig,pixels):
    mn=np.array(sig['rect_min']);mx=np.array(sig['rect_max']);tile=pixels//16
    m=((tile[None]>=mn[:,None])&(tile[None]<mx[:,None])).all(-1)
    assert m.tolist()==sig['selected_tile_members']
    return m


def math_checks():
    n1,n2,p1,p2,s1,s2,b,r,u=sp.symbols('n1 n2 p1 p2 s1 s2 b r u',real=True)
    phi=n1*(s1+(1-u)*(p1-s1))+n2*(s2+(1-u)*(p2-s2))+b
    reduced=sp.expand(phi-((1-u)*(n1*p1+n2*p2+b)+u*(n1*s1+n2*s2+b)))
    assert reduced==0
    G1,G2,B,c1,c2=sp.symbols('G1 G2 B c1 c2',real=True)
    contraction=-(n1*(G1-c1*B)+n2*(G2-c2*B)+(b+n1*c1+n2*c2)*B)
    assert sp.expand(contraction-(-n1*G1-n2*G2-b*B))==0
    # Signed floor and exclusive upper, including negative support coordinates.
    rng=np.random.default_rng(202610203);tests=0
    for _ in range(10000):
        lo=Q(int(rng.integers(-500000,500001)),int(rng.integers(1,1001)))
        hi=lo+Q(int(rng.integers(0,20001)),int(rng.integers(1,101)))
        L=lo//16;U=hi//16+1
        first=-((-lo.numerator)//lo.denominator);last=hi.numerator//hi.denominator
        if first<=last:assert L<=first//16 and last//16<U
        tests+=1
    return dict(homothetic_level_set_identity=True,anchor_contraction_identity=True,signed_rational_aabb_tests=tests,
        limitations='Real arithmetic support theorem and rational integer binning only; one-pixel floating guard is not a universal IEEE bound.')


def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    plan=json.loads((OUT/'input_plan.json').read_text());results=json.loads((OUT/'results.json').read_text())
    execution=json.loads((OUT/'execution_plan.json').read_text())
    assert results['input_plan_sha256']==sha(OUT/'input_plan.json')
    assert results['execution_plan_sha256']==sha(OUT/'execution_plan.json')
    assert execution['script_sha256']==sha(ROOT/'probe_tgs_stable_window.py')
    assert plan['protocol_sha256']==sha(ROOT/'TGS_STABLE_WINDOW_PROTOCOL_V2.md')
    oldmanifest=json.loads((ROOT/'experiments/tgs_radius_contraction_v1/evidence_manifest.json').read_text())
    for e in oldmanifest['files']:
        p=ROOT/e['path'];assert p.stat().st_size==e['size'] and sha(p)==e['sha256'],str(p)
    source=ROOT/'sources/tgs_native_diagnostic_v1'
    sm=json.loads((source/'manifest.json').read_text())
    for e in sm['files']:assert sha(source/e['path'])==e['sha256']
    for variant in ['direct','contraction']:
        receipt=json.loads((OUT/variant/'build_receipt.json').read_text())
        assert receipt['generator_sha256']==sha(ROOT/'tgs_stable_window_backend.py')
        assert receipt['module_sha256']==execution['modules'][variant]==sha(Path(receipt['module_file']))
        for e in receipt['sources']:assert sha(Path(e['path']))==e['sha256']
    assert (OUT/'direct/generated/forward.cu').read_bytes()==(OUT/'contraction/generated/forward.cu').read_bytes()
    cases=[(family,e) for family in ['old_cases','stress_cases','rational_cases','holdout_cases'] for e in plan[family]]
    rows=[];fdrows=[];record_index={(r['name'],r['variant']):r for r in results['records']}
    for family,e in cases:
        path=ROOT/e['input'];assert sha(path)==e['sha256']
        with np.load(path) as z:f={k:z[k].copy() for k in z.files}
        for k in ['n','w','h','degree']:f[k]=int(f[k])
        for k in ['name','kind','check']:f[k]=str(f[k])
        fun,grad=(aux_eval,aux_grad) if f['kind']=='aux' else (rgb_eval,rgb_grad)
        patterns=f.get('patterns',np.array([f['pixel_grad']]))
        references=[]
        for pattern in patterns:
            f['pixel_grad']=pattern;loss,samples,fp=fun(f['theta'].astype(float),f)
            references.append((samples,grad(f['theta'],f),fp))
        arrays={}
        for variant in ['direct','contraction']:
            rec=record_index[(f['name'],variant)]
            with np.load(OUT/(variant+'_'+f['name']+'.npz')) as z:a={k:z[k].copy() for k in z.files}
            arrays[variant]=a
            assert np.isfinite(a['native_rgb']).all() and np.isfinite(a['native_aux']).all() and np.isfinite(a['native_gradients']).all()
            assert np.array_equal(a['rect_min'],rec['base_signature']['rect_min']) and np.array_equal(a['rect_max'],rec['base_signature']['rect_max'])
            members(rec['base_signature'],f['pixels'])
            vg=[]
            for pi,(_,g,_) in enumerate(references):
                assert np.linalg.norm(g-a['reference_gradients'][pi])<=1e-8*max(np.linalg.norm(g),1)
                actual=a['native_gradients'][pi];n=f['n']
                assert np.array_equal(actual,np.array(rec['patterns'][pi]['native_gradient']))
                groups=dict(geometry=check_group(actual[:9*n],g[:9*n]),sigma_opacity=check_group(actual[9*n:11*n],g[9*n:11*n]))
                if f['degree']>=0:groups['SH']=check_group(actual[11*n:],g[11*n:])
                special=float(actual@f['special_direction']) if f['check']=='homothety_nullspace' else None
                gate=abs(special)<=1e-6 if special is not None else all(v['gate'] for v in groups.values())
                fe=float(np.abs(a['native_samples']-references[pi][0]).max())
                assert abs(fe-rec['patterns'][pi]['forward_max_abs'])<1e-12
                vg.append(dict(groups=groups,forward_max_abs=fe,null_direction=special,gate=bool(gate and fe<=3e-5 and (a['native_radii']>0).all())))
            allfd=True
            for d in rec['finite_differences']:
                pi=d['pattern'];f['pixel_grad']=patterns[pi];step=d['step'];vec=f['fd_directions'][pi,d['direction']]
                plus=(f['theta'].astype(float)+step*vec).astype(np.float32);minus=(f['theta'].astype(float)-step*vec).astype(np.float32)
                cp,_,fp=fun(plus.astype(float),f);cm,_,fm=fun(minus.astype(float),f)
                actual=(plus.astype(float)-minus.astype(float))/(2*step)
                assert np.array_equal(actual,d['actual_direction'])
                weight=f['aux_grad'] if f['kind']=='aux' else patterns[pi]
                native_fd=float(((np.array(d['plus_samples'])-np.array(d['minus_samples']))*weight).sum())/(2*step)
                cpu_fd=float(cp-cm)/(2*step);expected=float(references[pi][1]@actual)
                assert abs(native_fd-d['native_fd'])<1e-8*max(1,abs(native_fd))
                assert abs(cpu_fd-d['cpu_fd'])<1e-8*max(1,abs(cpu_fd))
                assert abs(expected-d['expected'])<1e-8*max(1,abs(expected))
                pm=members(d['plus_signature'],f['pixels']);mm=members(d['minus_signature'],f['pixels']);base=members(rec['base_signature'],f['pixels'])
                tile=bool(np.array_equal(pm,mm) and np.array_equal(pm,base))
                alpha=d['plus_signature']['alpha_support']==d['minus_signature']['alpha_support']==rec['base_signature']['alpha_support']
                kernel=fp==fm==references[pi][2]
                smooth=bool(tile and alpha and kernel);assert tile==d['tile_membership_stable'] and alpha==d['alpha_support_stable'] and smooth==d['smooth']
                allfd&=smooth
                fdrows.append(dict(name=f['name'],variant=variant,step=step,pattern=pi,direction=d['direction'],expected=expected,native_fd=native_fd,cpu_fd=cpu_fd,
                    error_normalized=abs(native_fd-expected)/max(abs(expected),1e-6),smooth=smooth))
            coverage=None
            if family=='holdout_cases':
                path=ROOT/e['selection'];assert sha(path)==e['selection_sha256']
                with np.load(path) as z:pix=z['pixels'];saved=z['ideal_valid']
                fc=dict(f,pixels=pix,pixel_grad=np.zeros((len(pix),3)))
                _,_,fp=rgb_eval(fc['theta'].astype(float),fc);valid=np.array(fp['valid'][0]);assert np.array_equal(valid,saved)
                tile=pix//16;mn=a['rect_min'][0];mx=a['rect_max'][0];binned=((tile>=mn)&(tile<mx)).all(-1)
                alpha=a['native_aux'][1,pix[:,1],pix[:,0]]
                coverage=dict(ideal_valid_samples=int(valid.sum()),tile_false_negatives=int((valid&~binned).sum()),native_support_false_negatives=int((valid&(alpha<=0)).sum()))
                assert coverage==rec['coverage']
            gate=all(v['gate'] for v in vg) and allfd and (coverage is None or (coverage['tile_false_negatives']==coverage['native_support_false_negatives']==0))
            rows.append(dict(name=f['name'],family=family,variant=variant,patterns=vg,coverage=coverage,all_fd_support_stable=allfd,gate=gate))
        same=all(np.array_equal(arrays['direct'][key],arrays['contraction'][key]) for key in ['native_rgb','native_aux','native_radii','rect_min','rect_max'])
        for row in rows[-2:]:row['forward_identical']=same;row['gate']=row['gate'] and same;assert row['gate']==record_index[(row['name'],row['variant'])]['gate']
    audit=dict(protocol_sha256=plan['protocol_sha256'],input_plan_sha256=sha(OUT/'input_plan.json'),execution_sha256=sha(OUT/'execution_plan.json'),results_sha256=sha(OUT/'results.json'),script_sha256=sha(Path(__file__)),math=math_checks(),old_manifest_files_unchanged=len(oldmanifest['files']),original_source_files_unchanged=len(sm['files']),cases=len(cases),vjp_count=sum(len(r['patterns']) for r in rows),fd_count=len(fdrows),rows=rows,
        all_gates=all(r['gate'] for r in rows),failed={k:[r['name'] for r in rows if r['variant']==k and not r['gate']] for k in ['direct','contraction']},
        max_forward_abs=max(p['forward_max_abs'] for r in rows for p in r['patterns']),max_geometry_relative=max(p['groups']['geometry']['relative'] for r in rows for p in r['patterns']),
        holdout_ideal_valid_samples_per_version=sum(r['coverage']['ideal_valid_samples'] for r in rows if r['variant']=='direct' and r['coverage'] is not None),
        max_native_fd_normalized_error=max(d['error_normalized'] for d in fdrows),all_fd_support_stable=all(d['smooth'] for d in fdrows),
        performance_measurement_allowed=all(r['gate'] for r in rows),training=False)
    np.savez_compressed(OUT/'independent_fd_audit.npz',expected=np.array([r['expected'] for r in fdrows]),native_fd=np.array([r['native_fd'] for r in fdrows]),cpu_fd=np.array([r['cpu_fd'] for r in fdrows]),error_normalized=np.array([r['error_normalized'] for r in fdrows]),step=np.array([r['step'] for r in fdrows]),smooth=np.array([r['smooth'] for r in fdrows]))
    target.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
