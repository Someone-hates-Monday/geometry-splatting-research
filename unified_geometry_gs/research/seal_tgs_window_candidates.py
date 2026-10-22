"""Seal v9 candidate-specific decisions and all precision/cost evidence without promotion."""
import ast,datetime,json
from pathlib import Path
from tgs_window_candidates_common import ROOT,OUT,sha

def read(name):return json.loads((OUT/name).read_text())

def local_dependencies(paths):
    found=set(paths);queue=[p for p in paths if p.suffix=='.py']
    while queue:
        p=queue.pop()
        for node in ast.walk(ast.parse(p.read_text())):
            names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else ([a.name for a in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                dep=ROOT/(name.split('.')[0]+'.py')
                if dep.is_file() and dep not in found:found.add(dep);queue.append(dep)
    return found

def main():
    target=OUT/'evidence_manifest.json';assert not target.exists()
    precision=read('precision_audit.json');cost=read('cost/independent_audit.json');census=read('failure_census.json');p=read('input_plan.json');freeze=read('audit_plan.json')
    assert precision['candidate_gates']=={'coarse':True,'standard':False} and not precision['all_gates']
    assert precision['performance_measurement_allowed']=={'coarse':True,'standard':False}
    assert cost['all_verified'] and cost['engineering_screen']=={'coarse':True} and cost['measured_pairs']==378
    assert cost['failures_excluded_only_by_precision']==['standard'] and not cost['scientific_goal_complete']
    assert census['rows'][4]['name']=='_DSC9213' and census['rows'][4]['candidates']['standard']['bad_pixel_count']==6
    assert all(r['candidates']['coarse']['bad_pixel_count']==0 for r in census['rows'])
    assert census['all_coarse_FD_sample_endpoints_bitwise_oracle'] and census['FD_pairs_compared']==2592
    assert sha(OUT/'input_plan.json')==freeze['input_plan_sha256']
    for name,h in freeze['scripts'].items():assert sha(ROOT/name)==h
    for name,h in p['sources'].items():assert sha(ROOT/name)==h
    for name,h in p['test_scripts'].items():assert sha(ROOT/name)==h
    for output,script in [('precision_audit.json','audit_tgs_window_candidates_precision.py'),('cache_audit.json','audit_tgs_window_candidates_cache.py'),('binary_audit.json','audit_tgs_window_candidates_binary.py'),('fd_results.json','probe_tgs_window_candidates_fd.py'),('failure_census.json','summarize_tgs_window_candidates_failures.py'),('cost/independent_audit.json','audit_tgs_window_candidates_cost.py')]:assert read(output)['script_sha256']==sha(ROOT/script)
    for name,h in read('cost_analysis_plan.json')['scripts'].items():assert sha(ROOT/name)==h
    verified_prior=0
    for folder,h in p['prior_manifests'].items():
        m=ROOT/'experiments'/folder/'evidence_manifest.json';assert sha(m)==h
        for en in json.loads(m.read_text())['files']:assert sha(ROOT/en['path'])==en['sha256'];verified_prior+=1
    assert verified_prior==precision['prior_hash_checks']
    for entry in json.loads((ROOT/'sources/tgs_native_diagnostic_v1/manifest.json').read_text())['files']:assert sha(ROOT/'sources/tgs_native_diagnostic_v1'/entry['path'])==entry['sha256']
    for suffix in ['', '_v2','_v3']:
        meta=read('cost/figure_metadata'+suffix+'.json');assert meta['analysis_sha256']==sha(OUT/'cost/analysis.json') and meta['script_sha256']==sha(ROOT/('plot_tgs_window_candidates_cost'+suffix+'.py'))
        for name,h in meta['files'].items():assert sha(OUT/'cost'/name)==h
    assert len({sha(OUT/'cost'/name) for name in ['figure_data.csv','figure_data_v2.csv','figure_data_v3.csv']})==1
    (OUT/'figure_visual_review.json').write_text(json.dumps(dict(reviewed='paired_cost_v3.png',all_camera_labels_and_axes_visible=True,all_raw_paired_blocks_shown=True,earlier_layouts=['paired_cost.png','paired_cost_v2.png'],earlier_layout_issue='Top title/legend/camera clipping due to layout rect height interpretation. Preserved as failed layouts only.',data_and_statistics_unchanged=True,figure_sha256=sha(OUT/'cost/paired_cost_v3.png')),indent=2)+'\n')
    decision=dict(precision={'coarse':True,'standard':False},engineering_screen={'coarse':True,'standard':None},standard_timing_allowed=False,standard_failure=dict(camera='_DSC9213',new_depth_bad_pixels=6,maximum=3.4332275390625e-5,threshold=3e-5,old14_now_pass=True),coarse_forward_bitwise_native64=True,production_promotion=False,training_authorized_in_this_phase=False,training_run=False,GT_quality_validated=False,cross_scene_validated=False,novelty_established=False,scientific_goal_complete=False,reason='Bounded same-semantic operator improvement is supported on one checkpoint/session; historical different-numerics piecewise remains faster. Full scientific evaluation remains outstanding.')
    (OUT/'promotion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    names=['TGS_WINDOW_CANDIDATES_PROTOCOL_V9.md','TGS_WINDOW_COST_PROTOCOL_V9.md','TGS_COARSE_REJECTION_NOTE_V9.md','TGS_WINDOW_CANDIDATES_RESULT_2026_10_20.md','tgs_coarse_exact_window_v9.cuh','tgs_compensated_window_v8.cuh','tgs_window_candidates_common.py','prepare_tgs_window_candidates.py','tgs_window_candidates_backend.py','probe_tgs_window_candidates_cases.py','probe_tgs_window_candidates_real.py','probe_tgs_window_candidates_fd.py','audit_tgs_window_candidates_binary.py','audit_tgs_window_candidates_cache.py','audit_tgs_window_candidates_precision.py','benchmark_tgs_window_candidates.py','analyze_tgs_window_candidates_cost.py','audit_tgs_window_candidates_cost.py','summarize_tgs_window_candidates_failures.py','plot_tgs_window_candidates_cost.py','plot_tgs_window_candidates_cost_v2.py','plot_tgs_window_candidates_cost_v3.py','seal_tgs_window_candidates.py']
    sources=local_dependencies({ROOT/name for name in names});files=sources|{q for q in OUT.rglob('*') if q.is_file() and '__pycache__' not in q.parts}
    entries=[dict(path=q.relative_to(ROOT).as_posix(),size=q.stat().st_size,sha256=sha(q)) for q in sorted(files)]
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries,source_dependency_files=len(sources),candidate_precision_decisions=precision['candidate_gates'],candidate_cost_screen=cost['engineering_screen'],prior_hash_checks=verified_prior,original_source_checks=precision['original_source_hash_checks'],training=False,scientific_goal_complete=False)
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    for en in entries:assert sha(ROOT/en['path'])==en['sha256']
    print(json.dumps(dict(files=len(entries),bytes=sum(e['size'] for e in entries),source_dependency_files=len(sources),manifest_sha256=sha(target),goal_active=True)),flush=True)

if __name__=='__main__':main()
