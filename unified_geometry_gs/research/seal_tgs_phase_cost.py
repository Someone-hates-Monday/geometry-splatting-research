"""Seal a correctness success and measured efficiency rejection together."""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_phase_cost_v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'evidence_manifest.json';assert not target.exists();prior=0
    for name in ['tgs_radius_contraction_v1','tgs_stable_window_v2','tgs_adaptive_window_v3','tgs_compositing_guard_v4']:
        for e in json.loads((ROOT/'experiments'/name/'evidence_manifest.json').read_text())['files']:
            p=ROOT/e['path'];assert p.stat().st_size==e['size'] and sha(p)==e['sha256'];prior+=1
    precision=json.loads((OUT/'precision_audit.json').read_text());cost=json.loads((OUT/'cost/independent_audit.json').read_text());assert precision['all_gates'] and cost['all_four_median_ratios_above_one'] and not cost['efficiency_promotion_supported']
    decision=dict(precision_scope_passed=True,efficiency_promotion_supported=False,production_promotion_allowed=False,training_performed=False,GT_quality_gain_proven=False,novelty_established=False,
        reason='All four observed paired guard/oracle median native-call ratios exceed one. Preserve correct arithmetic and measured negative cost result; do not promote the dual-pass/FP64-heavy prototype as acceleration.',
        precision_audit_sha256=sha(OUT/'precision_audit.json'),cost_audit_sha256=sha(OUT/'cost/independent_audit.json'),next='Verify compensated-dot/filter error theory, then version a lower-cost candidate with all retained gates and additional holdouts.')
    (OUT/'promotion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    (OUT/'pre_execution_notes.json').write_text(json.dumps(dict(initial_input_freeze_attempt='Windows imported Torch through a shared helper and failed before creating a plan or running CUDA',repair='Lazy-import GPU workload dependencies; freeze JSON-only inputs with the existing Windows NumPy runtime',native_compilation_or_execution_failure=False,training=False),indent=2)+'\n')
    names=['TGS_PHASE_COST_PROTOCOL_V5.md','TGS_PHASE_COST_RESULT_2026_10_20.md','tgs_phase_cost_backend.py','tgs_phase_common.py','probe_tgs_phase_cases.py','probe_tgs_phase_real.py','probe_tgs_phase_fd.py','audit_tgs_phase_binary.py','audit_tgs_phase_precision.py','benchmark_tgs_phase_cost.py','analyze_tgs_phase_cost.py','audit_tgs_phase_cost.py','plot_tgs_phase_cost.py','seal_tgs_phase_cost.py']
    files=[ROOT/n for n in names];files.extend(p for p in OUT.rglob('*') if p.is_file() and (p.suffix in ['.json','.npz','.csv','.cu','.cuh','.h','.cpp','.so','.png','.pdf','.txt'] or p.name=='build.ninja') and '__pycache__' not in p.parts)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))]
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=entries,file_count=len(entries),bytes=sum(e['size'] for e in entries),prior_hash_checks=prior,decision=decision,
        exclusions='Mutable CONTINUATION.md, object/compiler-cache files and this manifest. Includes generated sources, modules, SASS, raw samples, figures and metadata.',training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    for e in entries:assert sha(ROOT/e['path'])==e['sha256']
    print(json.dumps(dict(files=len(entries),bytes=result['bytes'],prior_hash_checks=prior,manifest_sha256=sha(target),efficiency_promotion_supported=False)),flush=True)
if __name__=='__main__':main()
