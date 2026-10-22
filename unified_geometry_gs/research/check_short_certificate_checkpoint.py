"""Terminal handles, artifacts and frozen snapshot audit for this goal stage."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=R/'results/short_certificate_checkpoint_v1.json';assert not out.exists()
    snap=R/'snapshots/after_short_certificate_v1.zip';manifest=read(snap.with_suffix('.manifest.json'))
    assert sha(snap)==manifest['archive_sha256']
    latex=read(R/'results/latex_compile_status_v29.json');assert latex['kind']=='success' and sha(R/'median_depth_note.tex')==latex['source_sha256']
    guard=read(R/'results/canvas_short_certificate_guard_v1.json')
    assert sha(Path(guard['destination']))==guard['staged_sha256'] and not read(R/'results/canvas_short_certificate_syntax_v1.json')['errors']
    summary=read(R/'experiments/short_certificate_v1/summary.json');audit=read(R/'results/short_certificate_audit_v1.json')
    assert summary['development_numerical_timing_gate_pass'] and audit['local_adoption_gate_pass']
    fbase=R/'experiments/short_fallback_v1';fmanifest=read(fbase/'manifest.json');fallback=read(fbase/'summary.json')
    assert fmanifest['status']==fallback['status']=='complete' and fallback['no_false_certificate_gate_pass'] and fallback['strict_success_branch_covered']
    for n,h in fmanifest['source_sha256'].items():assert sha(R.parent/n)==h
    for n,h in fmanifest['binary_sha256'].items():assert sha(R.parent/n)==h
    for n,h in fmanifest['artifacts_sha256'].items():assert sha(fbase/n)==h
    with zipfile.ZipFile(fbase/'sources.zip') as z:
        for n,h in fmanifest['source_sha256'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
    names=['research/short_certificate_cuda.py','research/audit_short_certificate.py','research/verify_short_certificate.py',
           'research/SHORT_CERTIFICATE_PROTOCOL.md','research/SHORT_CERTIFICATE_RESULT.md',
           'research/experiments/short_certificate_v1/summary.json','research/experiments/short_certificate_v1/outputs.npz',
           'research/results/short_certificate_audit_v1.json','research/probe_short_fallback.py',
           'research/experiments/short_fallback_v1/summary.json','research/median_depth_note.tex',
           'research/results/latex_compile_status_v29.json','research/canvas_result_snapshot.tsx']
    with zipfile.ZipFile(snap) as z:
        assert json.loads(z.read('SNAPSHOT_MANIFEST.json'))['files_sha256']==manifest['files_sha256']
        for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==manifest['files_sha256'][n]
    record=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,
        previous_goal_turn_classification='progress: isolated reciprocal caching passed bitwise equivalence and local speed gate',
        current_goal_turn_classification='progress: one short sufficient certificate with independent scalar/world checks, paired speed gate and successful strict fallback probe',
        main_source_rows=7322,finite_models=106,no_root_models=4,independent_MP240_scalar_checks=774,independent_MP240_world_endpoint_pairs=106,
        main_final_short=106,main_strict_attempts=30,main_final_high=30,
        paired_combined_event_ratio=summary['paired_event_ratio_geometric_mean']['combined'],
        control_combined_event_median_ms=summary['median_event_ms']['control']['combined'],
        variant_combined_event_median_ms=summary['median_event_ms']['variant']['combined'],
        separate_flat_probe=dict(models=8,short=3,strict=1,unresolved=4,no_false_certificates=True),
        terminal_handles=dict(main_session78420=0,independent_MP240_verifier_exit_code=0,separate_flat_probe_exit_code=0,snapshot_session67005=0),
        source_latex_matches_builtin_success=True,source_canvas_matches_syntax_checked_stage=True,
        canvas_syntax_checker_rows24_erratum='Historical constant; not tab or row count. SDK/runtime not verified.',
        snapshot=dict(sha256=manifest['archive_sha256'],files=len(manifest['files_sha256']),bytes=snap.stat().st_size,selected_checks=names),
        limits=['Observed fixed mathematical ray diagnostic, not new scene confirmation or full renderer/backward/GT/training evidence.',
                'Flat family still has4 unresolved cases; standard short series bounds do not establish academic novelty.'],
        next='Full-image cost and cache scalability, explicit unresolved behavior and independent world VJP before expanding training.')
    out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
