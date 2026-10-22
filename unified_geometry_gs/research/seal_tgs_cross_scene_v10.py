"""Seal new cross-scene evidence, its preserved failures/repairs and unchanged prior state."""
import ast,datetime,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import torch
from tgs_cross_scene_common_v10 import ROOT,OUT,SOURCE,V9,V9_SHA

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block=f.read(8*1024*1024)
            if not block:break
            h.update(block)
    return h.hexdigest()
def read(path):return json.loads(path.read_text())
def dependencies(paths):
    found=set(paths);queue=[p for p in found if p.suffix=='.py']
    while queue:
        p=queue.pop()
        for node in ast.walk(ast.parse(p.read_text())):
            names=[node.module] if isinstance(node,ast.ImportFrom) and node.module else ([x.name for x in node.names] if isinstance(node,ast.Import) else [])
            for name in names:
                dep=ROOT/(name.split('.')[0]+'.py')
                if dep.is_file() and dep not in found:found.add(dep);queue.append(dep)
    return found

def main():
    target=OUT/'evidence_manifest.json';assert not target.exists();precision=read(OUT/'precision_audit.json');cost=read(OUT/'cost/independent_audit.json');plan=read(OUT/'real_plan.json');inp=read(OUT/'input_audit.json');repair=read(OUT/'audit_repair_plan.json');real=read(OUT/'real_results.json')
    assert precision['all_gates'] and precision['performance_measurement_allowed'] and precision['literal_bits_audited'] and len(precision['rows'])==4 and real['all_gates']
    assert cost['all_verified'] and cost['engineering_screen']=={'coarse':True} and cost['measured_pairs']==216 and cost['balanced_blocks']==72
    assert precision['script_sha256']==repair['script_sha256']==sha(ROOT/'audit_tgs_cross_scene_precision_v10_v2.py')
    assert repair['original_auditor_sha256']==sha(ROOT/'audit_tgs_cross_scene_precision_v10.py') and repair['original_failure_sha256']==sha(OUT/'audit_failure.txt') and repair['diagnostic_sha256']==sha(OUT/'recapture_diagnostic.json')
    assert precision['audit_repair_plan_sha256']==sha(OUT/'audit_repair_plan.json') and read(OUT/'recapture_diagnostic.json')['only_invisible_projection_scratch_differs']
    assert precision['real_results_sha256']==sha(OUT/'real_results.json') and precision['plan_sha256']==real['plan_sha256']==sha(OUT/'real_plan.json') and inp['passed'] and plan['input_audit_sha256']==sha(OUT/'input_audit.json')
    for name,h in plan['scripts'].items():assert sha(ROOT/name)==h
    for en in plan['modules'].values():assert sha(Path(en['path']))==en['sha256']
    for name,h in read(OUT/'cost_analysis_plan.json')['scripts'].items():assert sha(ROOT/name)==h
    for filename in ['literal_bit_audit.json','v9_literal_bit_audit.json','v9_FD_sample_bit_audit.json']:
        q=read(OUT/filename);assert q['literal_bits_equal'];script='audit_tgs_fd_sample_bits_v10.py' if filename=='v9_FD_sample_bit_audit.json' else 'audit_tgs_literal_bits_v10.py';assert q['script_sha256']==sha(ROOT/script)
    assert sha(V9/'evidence_manifest.json')==V9_SHA;prior=read(V9/'evidence_manifest.json')['files']
    for en in prior:assert sha(ROOT/en['path'])==en['sha256']
    print(json.dumps(dict(stage='prior_files_verified',count=len(prior))),flush=True)
    original=ROOT/'sources/tgs_native_diagnostic_v1';original_files=read(original/'manifest.json')['files']
    for en in original_files:assert sha(original/en['path'])==en['sha256']
    source_manifest=read(SOURCE/'manifest.json');acq=read(SOURCE/'acquisition_audit.json');assert acq['complete_members']==10 and acq['manifest_sha256']==sha(SOURCE/'manifest.json') and acq['successful_cached_network_payload_bytes']<=650*1024*1024
    for en in source_manifest['files']:assert sha(ROOT/en['local'])==en['sha256']
    fm=read(OUT/'cost/figure_metadata.json');assert fm['script_sha256']==sha(ROOT/'plot_tgs_cross_scene_cost_v10.py') and fm['analysis_sha256']==sha(OUT/'cost/analysis.json') and fm['audit_sha256']==sha(OUT/'cost/independent_audit.json')
    for name,h in fm['files'].items():assert sha(OUT/'cost'/name)==h
    decision=dict(new_scenes=['flowers7000','room7000'],camera_count=4,reference_forward_literal_bits=True,full_RGB_depth_VJP_gate=True,visible_cache_exact_subset_edges=sum(r['cache']['checked_edges'] for r in precision['rows']),cross_scene_cost_engineering_screen=True,measured_pairs=216,training_run=False,GT_camera_identity_verified=False,GT_quality_evaluated=False,novelty_established=False,production_promotion=False,end_to_end_training_speedup_proved=False,scientific_goal_complete=False,prior_v9_manifest_sha256=V9_SHA,limits='Two purposefully size-selected scenes and one unlocked-clock session; numerically different historical piecewise remains faster. Keep full scientific objective active.')
    (OUT/'promotion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    (OUT/'runtime_info.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_cuda_build=torch.version.cuda,numpy=np.__version__,platform=platform.platform(),hardware_identity_source='cost/plan.json telemetry_start and all block telemetry, no additional GPU workload'),indent=2)+'\n')
    (OUT/'figure_visual_review.json').write_text(json.dumps(dict(file='cost/paired_cost.png',sha256=sha(OUT/'cost/paired_cost.png'),all_labels_axes_and_legend_visible=True,raw_primary_paired_points_retained=True,different_numerics_context_disclosed=True,no_journal_compliance_claim=True),indent=2)+'\n')
    names=['TGS_CROSS_SCENE_PROTOCOL_V10.md','TGS_CROSS_SCENE_COST_PROTOCOL_V10.md','TGS_CROSS_SCENE_INPUT_NOTE_V10.md','TGS_CROSS_SCENE_AUDIT_REPAIR_V10.md','TGS_CROSS_SCENE_STATUS_2026_10_20.md','TGS_CROSS_SCENE_RESULT_2026_10_20.md','extract_tgs_cross_scene_v10.py','audit_tgs_cross_scene_acquisition_v10.py','tgs_cross_scene_common_v10.py','audit_tgs_cross_scene_inputs_v10.py','probe_tgs_cross_scene_v10.py','audit_tgs_cross_scene_precision_v10.py','audit_tgs_cross_scene_precision_v10_v2.py','diagnose_tgs_cross_scene_recapture_v10.py','audit_tgs_literal_bits_v10.py','audit_tgs_fd_sample_bits_v10.py','benchmark_tgs_cross_scene_v10.py','analyze_tgs_cross_scene_cost_v10.py','audit_tgs_cross_scene_cost_v10.py','plot_tgs_cross_scene_cost_v10.py','summarize_tgs_cross_scene_telemetry_v10.py','seal_tgs_cross_scene_v10.py']
    source_files=dependencies({ROOT/name for name in names});data_files={p for directory in [OUT,SOURCE] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts};assert not any(p.suffix=='.partial' for p in data_files)
    extra={ROOT/'sources/tgs_pretrained_listing_v1'/name for name in ['files.json','zip_entries.json','range_probe.bin']};files=sorted(source_files|data_files|extra);entries=[]
    for i,p in enumerate(files):
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)))
        if i%200==199:print(json.dumps(dict(stage='new_files_hashed',count=i+1,total=len(files))),flush=True)
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries,source_dependency_files=len(source_files),prior_hash_checks=len(prior),original_source_checks=len(original_files),reference_equivalence=True,cross_scene_cost_engineering_screen=True,GT_quality_validated=False,novelty_established=False,training=False,scientific_goal_complete=False)
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    for en in entries:assert sha(ROOT/en['path'])==en['sha256']
    print(json.dumps(dict(files=len(entries),bytes=sum(en['size'] for en in entries),source_dependency_files=len(source_files),manifest_sha256=sha(target),goal_active=True)),flush=True)

if __name__=='__main__':main()
