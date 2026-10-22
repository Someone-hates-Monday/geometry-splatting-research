"""Seal v3 successes and remaining real auxiliary discrepancies together."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'evidence_manifest.json';assert not target.exists()
    previous=0
    for name in ['tgs_radius_contraction_v1','tgs_stable_window_v2']:
        for e in json.loads((ROOT/'experiments'/name/'evidence_manifest.json').read_text())['files']:
            p=ROOT/e['path'];assert p.stat().st_size==e['size'] and sha(p)==e['sha256'];previous+=1
    audit=json.loads((OUT/'independent_audit.json').read_text());real=json.loads((OUT/'real_independent_audit.json').read_text())
    assert audit['all_candidate_gates'] and not audit['fd_support_failures'] and real['all_declared_gates']
    assert not real['production_promotion_allowed'] and not real['performance_measurement_allowed']
    decision=dict(fixed_95_case_rgb_vjp_gates=True,fixed_real_rgb_vjp_gates=True,real_auxiliary_tail_followup_required=True,production_promotion_allowed=False,performance_measurement_allowed=False,training=False,
        reason='Retain the two weighted-depth outliers and nonidentical traversal counts; trace compositing and verify uncertainty handling before promotion.',
        independent_audit_sha256=sha(OUT/'independent_audit.json'),real_audit_sha256=sha(OUT/'real_independent_audit.json'))
    (OUT/'promotion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    (OUT/'generation_attempt_1.json').write_text(json.dumps(dict(stage='Before CUDA compiler invocation',outcome='AssertionError: once matched 2 occurrences of radii[idx] = 0;',preserved_partial_tree='generated_partial_1',repair='Use unique density_factor initialization as counter initialization anchor',original_generator_full_snapshot_preserved=False,native_execution_occurred=False),indent=2)+'\n')
    names=['TGS_ADAPTIVE_WINDOW_PROTOCOL_V3.md','TGS_ADAPTIVE_LOG_BOUND_NOTE.md','TGS_ADAPTIVE_WINDOW_RESULT_2026_10_20.md',
        'prepare_tgs_adaptive_inputs.py','tgs_adaptive_window.cuh','tgs_adaptive_window_backend.py','probe_tgs_adaptive_window.py',
        'tgs_barycentric_reference.py','probe_tgs_adaptive_fd.py','audit_tgs_adaptive_window.py','measure_tgs_adaptive_real.py',
        'audit_tgs_adaptive_real.py','seal_tgs_adaptive_window.py']
    files=[ROOT/n for n in names]
    files.extend(p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.csv','.cu','.h','.cuh','.cpp','.so'] and '__pycache__' not in p.parts)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=entries,file_count=len(entries),bytes=sum(e['size'] for e in entries),previous_hash_checks=previous,
        exclusions='Mutable CONTINUATION.md, compiler object/ninja/cache files and this manifest; CUDA modules and generated source/header trees, including partial attempt, are included.',promotion=decision,training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    for e in entries:assert sha(ROOT/e['path'])==e['sha256']
    print(json.dumps(dict(file_count=len(entries),bytes=result['bytes'],previous_hash_checks=previous,manifest_sha256=sha(target),performance_measurement_allowed=False,training=False)),flush=True)
if __name__=='__main__':main()
