"""Clone sources, adapt inherited tests and freeze the two-candidate cohort before output."""
import hashlib,json,shutil
from pathlib import Path
from tgs_window_candidates_common import ROOT,OUT,sha
def once(t,a,b):assert t.count(a)==1,(a,t.count(a));return t.replace(a,b)
def main():
    assert not OUT.exists();OUT.mkdir();v5=ROOT/'experiments/tgs_phase_cost_v5';v8=ROOT/'experiments/tgs_compensated_renderer_v8';power=ROOT/'experiments/tgs_standard_pow_dispatch_v1';expected={v8:'9111959a56dcabceba9ccebac94df3e42739998a8e51375a7a20e5513a16168f',power:'8a3a239b4fa7a967c32cc216f86183eddf4d1f7ec6eeaf8d671cc3e8f6fd8b10'}
    for folder,h in expected.items():
        assert sha(folder/'evidence_manifest.json')==h
        for e in json.loads((folder/'evidence_manifest.json').read_text())['files']:assert sha(ROOT/e['path'])==e['sha256']
    for family,parent in [('coarse',v5),('standard',v8)]:
        dst=OUT/family/'generated/native';shutil.copytree(parent/'generated/native',dst);cu=dst/'cuda_rasterizer'
        if family=='coarse':
            (cu/'stable_window.cuh').write_text((ROOT/'tgs_coarse_exact_window_v9.cuh').read_text());p=cu/'forward.cu';t=p.read_text();lines=t.splitlines(True);removed=[l for l in lines if 'stable_error[10*idx+' in l];assert len(removed)==4;t=''.join(l for l in lines if 'stable_error[10*idx+' not in l);t=t.replace('stable_error+10*','stable_error+14*');t=once(t,'rect_min[idx] = rect_min_triangle_test;','rect_min[idx] = rect_min_triangle_test;\n    coarse_precompute(stable_error+14*idx,stable_coeff+10*idx,center_triangle_2D,rect_min_triangle_test,rect_max_triangle_test,W,H);');t=once(t,'if(STABLE_MODE==2) {','if(false) { // coarse: no preflight; all surviving alpha evaluations are native64\n');p.write_text(t)
            p=cu/'backward.cu';t=p.read_text();assert 'stable_error+10*' in t;p.write_text(t.replace('stable_error+10*','stable_error+14*'));p=cu/'rasterizer_impl.cu';p.write_text(once(p.read_text(),'obtain(chunk, geom.stable_error, P*10, 128);','obtain(chunk, geom.stable_error, P*14, 128);'))
        else:
            p=cu/'stable_window.cuh';t=p.read_text();t=once(t,'struct StableWindow','extern "C" __device__ float __nv_powf(float,float);\nstruct StableWindow');t=once(t,'__powf(fmaxf(z.u,0.f),sigma)','__nv_powf(fmaxf(z.u,0.f),sigma)');p.write_text(t)
    # Same physical references, fixture masks and group tests. Expand only variant names/readers.
    t=(ROOT/'probe_tgs_compensated_cases.py').read_text();t=t.replace('from tgs_compensated_common import','from tgs_window_candidates_common import').replace('ROOT,OUT,VARIANTS,sha,modules','ROOT,OUT,VARIANTS,BASES,sha,modules,strict_groups').replace('TGS_COMPENSATED_RENDERER_PROTOCOL_V8.md','TGS_WINDOW_CANDIDATES_PROTOCOL_V9.md').replace("for v in ['guard','oracle']:","for v in BASES:").replace("decode(forward[1],f['n'])","decode(forward[1],f['n'],v)").replace('state(forward[1],f)','state(forward[1],f,v)')
    anchor='        for row in case:'
    insert='''        for v in ['coarse','coarse_diag']:
            a=data[v];eq=all(np.array_equal(a[k],oracle[k]) for k in ['native_rgb','native_aux','native_radii','maxblend','contributors'])
            gs=[groups(x,y,f['n'],f['degree']) for x,y in zip(a['native_gradients'],oracle['native_gradients'])]
            strict=all(strict_groups(list(g.values())) for g in gs)
            nativechecks[v]['coarse_exact_gate']=bool(eq and strict and not a['flags'].any() and not a['preflight'].any())
            nativechecks[v]['gate'] &= nativechecks[v]['coarse_exact_gate']
'''
    t=once(t,anchor,insert+anchor);t=once(t,"    (OUT/'case_results.json').write_text","    result['candidate_gates']={v:all(r['gate'] for r in rows if r['variant'] in [v,v+'_diag']) for v in BASES}\n    (OUT/'case_results.json').write_text");p=ROOT/'probe_tgs_window_candidates_cases.py';assert not p.exists();p.write_text(t)
    t=(ROOT/'probe_tgs_compensated_fd.py').read_text().replace('from tgs_compensated_common import','from tgs_window_candidates_common import').replace('ROOT,OUT,sha,modules','ROOT,OUT,BASES,sha,modules').replace("['all_gates']","['candidate_gates']['oracle']").replace("for v in ['guard','oracle']:","for v in BASES:");p=ROOT/'probe_tgs_window_candidates_fd.py';assert not p.exists();p.write_text(t)
    plan=json.loads((v8/'input_plan.json').read_text());cams=plan['cameras'];cp=ROOT/'sources/tgs_real_checkpoint_v1/stump/cameras.json';tests=sorted(json.loads(cp.read_text()),key=lambda c:c['img_name'])[::8];used={c['img_name'] for c in cams};new=min([c for c in tests if c['img_name'] not in used],key=lambda c:hashlib.sha256(('tgs_window_candidates_v9/'+c['img_name']).encode()).hexdigest());cams=cams+[dict(new,split='test',heldout_for_v9=True)]
    sources={p.relative_to(ROOT).as_posix():sha(p) for family in ['coarse','standard'] for p in (OUT/family/'generated/native').rglob('*') if p.is_file()};scripts=['tgs_window_candidates_common.py','probe_tgs_window_candidates_cases.py','probe_tgs_window_candidates_real.py','probe_tgs_window_candidates_fd.py','tgs_coarse_exact_window_v9.cuh']
    plan.update(protocol_sha256=sha(ROOT/'TGS_WINDOW_CANDIDATES_PROTOCOL_V9.md'),common_sha256=sha(ROOT/'tgs_window_candidates_common.py'),prepare_sha256=sha(Path(__file__)),cameras=cams,camera_source_sha256=sha(cp),sources=sources,test_scripts={p:sha(ROOT/p) for p in scripts},prior_manifests={f.name:h for f,h in expected.items()},variants=['coarse_diag','standard_diag','oracle_diag','coarse','standard','oracle'],training=False,timing=False)
    (OUT/'input_plan.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(dict(cases=len(plan['cases']),cameras=[c['img_name'] for c in cams],new_holdout=new['img_name'],input_plan_sha256=sha(OUT/'input_plan.json'))),flush=True)
if __name__=='__main__':main()
