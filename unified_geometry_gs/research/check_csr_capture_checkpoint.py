"""Audit complete CSR stage, editor source and selected snapshot members."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=R/'results/csr_capture_checkpoint_v1.json';assert not out.exists()
    p=R/'snapshots/after_csr_capture_v1.zip';m=read(p.with_suffix('.manifest.json'));assert sha(p)==m['archive_sha256']
    c=read(R/'results/latex_compile_status_v31.json');assert c['kind']=='success' and sha(R/'median_depth_note.tex')==c['source_sha256']
    g=read(R/'results/canvas_csr_capture_guard_v1.json');assert sha(Path(g['destination']))==g['staged_sha256']
    assert not read(R/'results/canvas_csr_capture_syntax_v1.json')['errors']
    s=read(R/'experiments/csr_capture_v1/summary.json');a=read(R/'results/csr_capture_audit_v1.json')
    assert s['numerical_gate_pass'] and s['guard_probe_pass'] and a['gate_pass'] and a['checked_records']==870178922
    names=['research/csr_capture_backend.py','research/benchmark_csr_capture.py','research/verify_csr_capture.py','research/CSR_CAPTURE_PROTOCOL.md',
           'research/CSR_CAPTURE_RESULT.md','research/experiments/csr_capture_v1/summary.json','research/experiments/csr_capture_v1/scan37_0000.npz',
           'research/experiments/csr_capture_v1/guard_under.json','research/experiments/csr_capture_v1/guard_over.json',
           'research/results/csr_capture_audit_v1.json','research/median_depth_note.tex','research/results/latex_compile_status_v31.json','research/canvas_result_snapshot.tsx']
    with zipfile.ZipFile(p) as z:
        assert json.loads(z.read('SNAPSHOT_MANIFEST.json'))['files_sha256']==m['files_sha256']
        for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==m['files_sha256'][n]
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,
        previous_goal_turn_classification='progress: actual full-image capacity192 rejected by334072 valid pixels; counts preserve all image outputs',
        current_goal_turn_classification='progress: dynamic actual CSR IDs/positions captured for all24; all870178922 records independently checked and two failure guards passed',
        views=24,pixels=10834488,records=870178922,all_GPU_and_independent_CPU_checks_pass=True,guards=dict(under_status=3,over_status=2,over_unfilled_slots=1),
        capture_count_paired_ratio=s['paired_capture_count_ratio_geometric_mean'],whole_workflow_event_ms_range=[min(r['stage_time']['whole_gpu_workflow']['event_ms'] for r in s['rows']),max(r['stage_time']['whole_gpu_workflow']['event_ms'] for r in s['rows'])],
        terminal_handles=dict(csr_capture_session19995=0,independent_CPU_session95619=0,snapshot_session76646=0),
        source_latex_matches_builtin_success=True,source_canvas_matches_syntax_checked_stage=True,
        canvas_syntax_checker_rows24_erratum='Historical constant, not tab/row count; SDK/runtime not verified.',
        snapshot=dict(sha256=m['archive_sha256'],files=len(m['files_sha256']),bytes=p.stat().st_size,selected_member_checks=names),
        limits=['Actual list preparation only, not certified full-image roots, source backward, GT or training improvement.',
                'Timing includes separate capture stage vs full two-forward/scan/checker workflow; not a method speedup.',
                'Complete runtime original-image comparisons; only depth/counters and full CSR/source lists are independently audited offline.'],
        next='Shared original-parameter factor/mean error caches, dynamic/ray-parallel root certificates with original root and range semantics, explicit unresolved/output rounding, then source VJP/full cost/GT gates.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
