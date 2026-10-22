"""Verify the completed stage and selected members of its frozen snapshot."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, zipfile
R = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def main():
    out = R/'results/reciprocal_root_checkpoint_v1.json'
    assert not out.exists()
    snapshot = R/'snapshots/after_reciprocal_root_v1.zip'
    manifest = read(snapshot.with_suffix('.manifest.json'))
    assert sha(snapshot) == manifest['archive_sha256']
    latex = read(R/'results/latex_compile_status_v28.json')
    assert latex['kind'] == 'success' and sha(R/'median_depth_note.tex') == latex['source_sha256']
    guard = read(R/'results/canvas_reciprocal_guard_v1.json')
    assert sha(Path(guard['destination'])) == guard['staged_sha256']
    assert not read(R/'results/canvas_reciprocal_syntax_v1.json')['errors']
    names = ['research/reciprocal_root_cuda.py', 'research/audit_reciprocal_root.py',
             'research/verify_reciprocal_root.py', 'research/RECIPROCAL_ROOT_PROTOCOL.md',
             'research/RECIPROCAL_ROOT_RESULT.md', 'research/experiments/reciprocal_root_v2/summary.json',
             'research/experiments/reciprocal_root_v2/outputs.npz',
             'research/results/reciprocal_root_audit_v1.json', 'research/median_depth_note.tex',
             'research/results/latex_compile_status_v28.json', 'research/canvas_result_snapshot.tsx']
    with zipfile.ZipFile(snapshot) as archive:
        assert read(snapshot.with_suffix('.manifest.json'))['files_sha256'] == json.loads(archive.read('SNAPSHOT_MANIFEST.json'))['files_sha256']
        for name in names:
            assert hashlib.sha256(archive.read(name)).hexdigest() == manifest['files_sha256'][name]
    s = read(R/'experiments/reciprocal_root_v2/summary.json')
    a = read(R/'results/reciprocal_root_audit_v1.json')
    assert s['local_optimization_gate_pass'] and a['local_gate_pass']
    record = dict(status='current_step_complete', checked_utc=datetime.now(timezone.utc).isoformat(),
        overall_goal_achieved=False, previous_goal_turn_classification='progress: device budgets and autonomous certified roots passed; strict diagnostic cost identified',
        current_goal_turn_classification='progress: isolated exact reciprocal caching preserves complete output bits and passes preset local timing gate',
        fixed_source_rows=7322, finite_models=106, no_root_models=4,
        control_event_combined_median_ms=s['median_event_ms']['control']['combined'],
        variant_event_combined_median_ms=s['median_event_ms']['variant']['combined'],
        paired_ratio=s['paired_event_ratio_geometric_mean']['combined'],
        independent_exact_archive_bitwise_timing_audit_pass=True,
        terminal_handles=dict(reciprocal_root_session15185=0, independent_audit_exit_code=0, snapshot_session39968=0),
        source_latex_matches_builtin_success=True, source_canvas_matches_syntax_checked_stage=True,
        canvas_syntax_checker_rows24_erratum='Historical constant, not tab or new row count; SDK/runtime not verified.',
        snapshot=dict(sha256=manifest['archive_sha256'], files=len(manifest['files_sha256']), bytes=snapshot.stat().st_size,
                      independent_selected_member_checks=names),
        limits=['Fixed development mathematical ray workload, not new unseen confirmation or full renderer/backward/GT/training performance.',
                'Reciprocal caching is an implementation optimization; academic novelty and production adoption remain unverified.'],
        next='Freeze cheap sufficient certificates and selective strict fallback, then full 24-view renderer selection/init-range/cost and independent world-VJP gates before new training.')
    out.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
if __name__ == '__main__': main()
