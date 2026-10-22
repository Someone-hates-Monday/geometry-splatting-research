"""Seal isolated-window evidence and all exact-endpoint three-pixel mechanism records."""
import ast,csv,datetime,json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_window_scene_v15';TRACE=OUT/'trace';OLD=ROOT/'experiments/tgs_split_color_v14'

def read(p):return json.loads(p.read_text())

def deps(paths):
    result=set(paths);queue=[p for p in result if p.suffix=='.py']
    while queue:
        p=queue.pop()
        for n in ast.walk(ast.parse(p.read_text())):
            names=[n.module] if isinstance(n,ast.ImportFrom) and n.module else [x.name for x in n.names] if isinstance(n,ast.Import) else []
            for name in names:
                q=ROOT/(name.split('.')[0]+'.py')
                if q.is_file() and q not in result:result.add(q);queue.append(q)
    return result

def main():
    target=OUT/'evidence_manifest.json';assert not target.exists()
    plan=read(OUT/'plan.json');res=read(OUT/'results.json');audit=read(OUT/'independent_audit.json')
    assert res['forwards']==64 and len(res['cases'])==32 and res['all_radii_and_white_controls']
    assert audit['all_saved_arrays_and_controls_verified'] and audit['cases']==32
    assert audit['results_sha256']==sha(OUT/'results.json') and audit['plan_sha256']==sha(OUT/'plan.json')
    assert audit['script_sha256']==sha(ROOT/'audit_tgs_window_scene_v15.py')
    assert res['script_sha256']==sha(ROOT/'render_tgs_window_scene_v15.py') and res['plan_sha256']==sha(OUT/'plan.json')
    for n,h in plan['scripts'].items():assert sha(ROOT/n)==h
    for n,h in plan['old_inputs'].items():assert sha(OLD/n)==h
    assert plan['protocol_sha256']==sha(ROOT/'TGS_WINDOW_SCENE_PROTOCOL_V15.md')
    assert sum(s['zero_isolated_change_cases'] for s in audit['summary'])==13
    assert sum(s['conditional_fixed_slot_bound_violation_cases'] for s in audit['summary'])==2
    assert sum(s['conditional_fixed_slot_bound_violation_pixels'] for s in audit['summary'])==3
    violations=read(OUT/'violation_pixels.json');assert len(violations['pixels'])==3
    assert violations['source_audit_sha256']==sha(OUT/'independent_audit.json')
    assert violations['script_sha256']==sha(ROOT/'freeze_tgs_window_violations_v15.py')
    tp=read(TRACE/'plan.json');tr=read(TRACE/'results.json');ta=read(TRACE/'independent_audit.json');build=read(TRACE/'build_receipt.json')
    assert tr['full_forwards']==3 and tr['trace_calls']==18 and tr['all_native_fields_literal'] and tr['all_primary_endpoints_literal']
    assert tr['plan_sha256']==sha(TRACE/'plan.json') and tr['script_sha256']==sha(ROOT/'trace_tgs_window_scene_v15.py')
    for p,h in tp['inputs'].items():assert sha(ROOT/p)==h
    assert ta['all_verified'] and ta['endpoint_count']==6 and ta['exact_recorded_alpha_decompositions']
    assert ta['results_sha256']==sha(TRACE/'results.json') and ta['plan_sha256']==sha(TRACE/'plan.json')
    assert ta['script_sha256']==sha(ROOT/'audit_tgs_window_trace_v15.py')
    assert ta['RN_helper_sha256']==sha(ROOT/'audit_tgs_compensated_depth_trace.py') and ta['transmittance_RN_checks']==1698
    assert all(r['no_early_stop_any_mode'] and r['all_external_packets_literal_identical'] and r['all_child_alphas_unchanged_by_reorder'] for r in ta['rows'])
    assert sha(Path(build['path']))==build['sha256'] and build['source_sha256']==sha(ROOT/'tgs_window_scene_trace_v15.cu')
    crossing=read(TRACE/'active_crossings.json');assert crossing['exact_centroid_and_swap_checks']
    assert crossing['script_sha256']==sha(ROOT/'analyze_tgs_order_crossings_v15.py') and crossing['trace_audit_sha256']==sha(TRACE/'independent_audit.json')
    assert [r['active_crossing_pairs'] for r in crossing['rows']]==[8,19,15]
    fm=read(OUT/'figure_metadata.json');assert fm['script_sha256']==sha(ROOT/'plot_tgs_window_scene_v15.py')
    assert fm['audit_sha256']==sha(OUT/'independent_audit.json') and fm['trace_audit_sha256']==sha(TRACE/'independent_audit.json')
    for p,h in fm['files'].items():assert sha(OUT/p)==h
    for name,count in [('window_scene_cases.csv',32),('window_scene_trace.csv',3)]:
        with (OUT/name).open() as f:assert sum(1 for _ in csv.DictReader(f))==count
    previous=OLD/'evidence_manifest.json';assert sha(previous)=='8b7736cf7e650cc960a23c19d3660f1db0fde4ab51480321404465ee9aed5411'
    write_new(OUT/'figure_visual_review.json',dict(file='window_scene.png',sha256=sha(OUT/'window_scene.png'),
        all32_cases_and3_pixels_retained=True,zero_overlap_disclosed=True,symlog_and_log_axes_disclosed=True,
        labels_and_legend_visible=True,limited_mechanism_not_quality_claim=True))
    write_new(OUT/'research_direction_decision.json',dict(native_forwards=67,trace_calls=18,
        window_redistribution_present=True,ordering_mechanism_verified_on_frozen_pixels=3,
        early_stop_on_trace_pixels=False,whole_image_ordering_attribution=False,
        method_novelty_established=False,GT_quality_gain=False,geometry_gain=False,efficiency_gain=False,
        training=False,scientific_goal_complete=False,
        next='Full-image physical-depth sorting baseline and GT quality-cost comparison against center and parent-slot order; existing 2DTS/other sorting methods are necessary prior art.'))
    scripts=list(ROOT.glob('*tgs_window*scene*v15*.py'))+[ROOT/'freeze_tgs_window_violations_v15.py',ROOT/'analyze_tgs_order_crossings_v15.py']
    docs=[ROOT/n for n in ['TGS_WINDOW_SCENE_PROTOCOL_V15.md','TGS_FIXED_SLOT_WINDOW_NOTE_V15.md','TGS_WINDOW_TRACE_ADDENDUM_V15.md',
        'TGS_PLANAR_SPLIT_ORDER_NOTE_V15.md','TGS_WINDOW_PRIOR_WORK_V15.md','TGS_WINDOW_SCENE_RESULT_2026_10_21.md']]
    source=deps(scripts+docs+[ROOT/'tgs_window_scene_trace_v15.cu'])
    refs={previous,ROOT/'literature/window_scene_prior_work_v15.json',Path(plan['module']['path']),ROOT/plan['module']['receipt']}
    for file in ['LICENSE.md','LICENSE_GS.md']:
        p=ROOT/'sources/tgs_native_diagnostic_v1/native'/file
        if p.is_file():refs.add(p)
    refs|={ROOT/p for p in tp['inputs'] if not (ROOT/p).is_relative_to(OUT)}
    # Referenced v14 inputs are individually verified above and recorded by hash; avoid duplicating all legacy files.
    files=sorted(source|refs|{p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in files]
    write_new(target,dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries,
        source_dependency_files=len(source),readonly_v14_input_references=plan['old_inputs'],
        prior_full_tree_rehashed=False,native_forwards=67,observer_calls=18,
        all6_native_endpoints_literal=True,three_pixel_order_mechanism=True,
        training=False,scientific_goal_complete=False))
    for en in entries:assert sha(ROOT/en['path'])==en['sha256']
    print(json.dumps(dict(files=len(entries),bytes=sum(e['size'] for e in entries),source_dependency_files=len(source),
        manifest_sha256=sha(target),scientific_goal_complete=False)),flush=True)

if __name__=='__main__':main()
