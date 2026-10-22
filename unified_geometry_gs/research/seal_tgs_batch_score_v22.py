"""Seal exact-score evidence, retaining original timer failure and separate recovery."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_batch_score_v22'

def main():
    assert not (OUT/'evidence_manifest.json').exists()
    v=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text());a=json.loads((OUT/'timing/independent_analysis.json').read_text())
    assert v['all_gates_passed'] and v['full_forwards']==20 and len(v['rows'])==16 and v['accepted_mutations']==0
    assert audit['all_passed'] and len(audit['rows'])==16 and (audit['negative_scores'],audit['positive_scores'],audit['zero_scores'])==(3,13,0)
    assert json.loads((OUT/'fixtures.json').read_text())['all_passed']
    for row in v['rows']:
        assert all(row['checks'].values()) and row['all_tile_lists_literal'] and row['visible_cache_fields_literal']
        assert row['delta_SSE']==row['batch_score']==row['full_gpu_score'] and row['debug_hashes']==row['reference_hashes']
    assert a['all_verified'] and a['blocks']==64 and a['measured_endpoints']==256 and a['actual_timing_forwards']==982 and a['original_run_exit_code']==1
    assert a['original_setup_and_final_telemetry_missing'] and a['no_GPU_timing_rerun'] and not (OUT/'timing/results.json').exists()
    setup=json.loads((OUT/'timing/setup_addendum.json').read_text());assert setup['native_forwards']==2 and len(setup['rows'])==2
    assert next(r for r in a['rows'] if r['K']==8 and r['scene']=='flowers')['estimate_given_original_PNG_no_geometry_cache']<1
    dep=set()
    for name in ['plan.json','timing/plan.json','timing/recovery_plan.json','timing/setup_addendum_plan.json','timing/independent_analysis.json']:
        p=json.loads((OUT/name).read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        dep.update(p['inputs'])
    assert sha(ROOT/'audit_tgs_batch_score_v22.py')==audit['source_sha256']
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'] and sha(ROOT/'tgs_batch_score_v22.cu')==b['source_sha256']
    ctrl=json.loads((OUT/'plan.json').read_text())['control'];assert sha(Path(ctrl['path']))==ctrl['sha256']
    dep.add(str(Path(ctrl['path']).relative_to(ROOT)))
    dep.update(['audit_tgs_batch_score_v22.py','TGS_BATCH_SCORE_CORRECTNESS_NOTE_V22.md','plot_tgs_batch_score_v22.py','seal_tgs_batch_score_v22.py','TGS_BATCH_SCORE_RESULT_2026_10_21.md'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file())
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,training=False,goal_complete=False,accepted_mutations=0,
        exact_real_proposals=16,original_timer_exit_code=1,planned_timing_forwards=954,actual_timing_forwards=982,separate_setup_forwards=2,
        phase_total_native_forwards=1004,timing_recovered_without_GPU_rerun=True,original_setup_unavailable=True,
        standalone_quality_or_policy_improvement=False,novelty_established=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
