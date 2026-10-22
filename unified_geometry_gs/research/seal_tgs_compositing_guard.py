"""Seal trace causality, successful guard tests and remaining precision/cost limits."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_compositing_guard_v4';TRACE=ROOT/'experiments/tgs_compositing_trace_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'evidence_manifest.json';assert not target.exists();previous=0
    for name in ['tgs_radius_contraction_v1','tgs_stable_window_v2','tgs_adaptive_window_v3']:
        for e in json.loads((ROOT/'experiments'/name/'evidence_manifest.json').read_text())['files']:
            p=ROOT/e['path'];assert p.stat().st_size==e['size'] and sha(p)==e['sha256'];previous+=1
    audit=json.loads((OUT/'independent_audit.json').read_text());assert audit['all_precision_gates'] and not audit['fd_support_failures']
    trace=json.loads((TRACE/'results.json').read_text());assert trace['all_endpoints_bitwise_equal']
    decision=dict(correctness_scope_passed=True,scope='127 fixed inputs plus two complete real cameras, stated native-oracle/CPU gates and fixed RGB/depth VJPs',
        arbitrary_precision_real_threshold_equivalence_proven=False,power_global_enclosure_proven=False,phase_consistency_review_remaining=True,
        noninstrumented_cost_validation_remaining=True,training_performed=False,GT_quality_gain_proven=False,novelty_established=False,
        current_instrumented_performance_measurement_allowed=False,production_promotion_allowed=False,
        next='Review explicit phase rounding, freeze non-counter cost protocol and retain all precision/FD/real gates; validate cost reduction on independent held-out cameras.',
        independent_audit_sha256=sha(OUT/'independent_audit.json'),diagnostic_summary_sha256=sha(OUT/'diagnostic_summary.json'))
    (OUT/'promotion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    (TRACE/'execution_notes.json').write_text(json.dumps(dict(initial_CPU_audit_attempt='Windows could not dereference the recorded Linux module path; no audit output was created',repair='Resolve the existing module via trace build directory plus recorded filename and verify SHA256',final_audit_exit=0,native_failures=False,training=False),indent=2)+'\n')
    names=['trace_tgs_compositing.py','audit_tgs_compositing_trace.py','TGS_COMPOSITING_GUARD_PROTOCOL_V4.md','prepare_tgs_guard_inputs.py',
        'TGS_COMPOSITING_INPUT_ADDENDUM_V4_1.md','extend_tgs_guard_inputs.py','tgs_compositing_guard_backend.py','probe_tgs_compositing_guard.py',
        'measure_tgs_guard_real.py','probe_tgs_guard_fd.py','audit_tgs_compositing_guard.py','summarize_tgs_guard.py',
        'TGS_COMPOSITING_INTERVAL_NOTE.md','TGS_COMPOSITING_GUARD_RESULT_2026_10_20.md','seal_tgs_compositing_guard.py']
    files=[ROOT/name for name in names]
    for base in [OUT,TRACE]:files.extend(p for p in base.rglob('*') if p.is_file() and (p.suffix in ['.json','.npz','.cu','.cuh','.h','.cpp','.so'] or p.name=='build.ninja') and '__pycache__' not in p.parts)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=entries,file_count=len(entries),bytes=sum(e['size'] for e in entries),previous_sealed_hash_checks=previous,
        exclusions='Mutable CONTINUATION.md, object files and compiler caches; generated sources, modules and Ninja build commands are included. This self-referential manifest excluded.',decision=decision,training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    for e in entries:assert sha(ROOT/e['path'])==e['sha256']
    print(json.dumps(dict(file_count=len(entries),bytes=result['bytes'],previous_sealed_hash_checks=previous,manifest_sha256=sha(target),training=False)),flush=True)
if __name__=='__main__':main()
