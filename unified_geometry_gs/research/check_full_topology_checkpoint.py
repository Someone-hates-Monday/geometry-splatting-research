"""Audit completed full24 capacity stage and its recoverable snapshot."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=R/'results/full_topology_checkpoint_v1.json';assert not out.exists()
    snapshot=R/'snapshots/after_full_topology_v1.zip';manifest=read(snapshot.with_suffix('.manifest.json'))
    assert sha(snapshot)==manifest['archive_sha256']
    latex=read(R/'results/latex_compile_status_v30.json');assert latex['kind']=='success' and sha(R/'median_depth_note.tex')==latex['source_sha256']
    guard=read(R/'results/canvas_full_topology_guard_v1.json')
    assert sha(Path(guard['destination']))==guard['staged_sha256'] and not read(R/'results/canvas_full_topology_syntax_v1.json')['errors']
    s=read(R/'experiments/full_topology_v1/summary.json');a=read(R/'results/full_topology_audit_v1.json')
    assert s['status']=='complete' and s['all_prespecified_24_views'] and s['all_original_outputs_bitwise'] and s['numerical_gate_pass']
    assert a['status']=='complete' and a['all24_views'] and a['all_counts_and_histograms_recomputed'] and a['source_isolation_pass']
    assert not a['capacity192_covers_all_initial'] and s['valid_depth_counts']['capacity_exceed']['192']['pixels']==334072
    names=['research/full_topology_backend.py','research/benchmark_full_topology.py','research/verify_full_topology.py',
           'research/FULL_TOPOLOGY_PROTOCOL.md','research/FULL_TOPOLOGY_RESULT.md',
           'research/experiments/full_topology_v1/summary.json','research/experiments/full_topology_v1/scan63_0041.npz',
           'research/results/full_topology_audit_v1.json','research/results/full_topology_audit_failure_v1.json',
           'research/median_depth_note.tex','research/results/latex_compile_status_v30.json','research/canvas_result_snapshot.tsx']
    with zipfile.ZipFile(snapshot) as z:
        assert json.loads(z.read('SNAPSHOT_MANIFEST.json'))['files_sha256']==manifest['files_sha256']
        for name in names:assert hashlib.sha256(z.read(name)).hexdigest()==manifest['files_sha256'][name]
    record=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,
        previous_goal_turn_classification='progress: short sufficient certificates, independent world/scalar validation and strict/unresolved probe passed',
        current_goal_turn_classification='progress: all24 actual native images instrumented without output changes; full capacity192 rejected by334072 valid pixels; independent audit complete',
        views=24,pixels=10834488,initial_eligible_pixels=9165957,final_valid_pixels=9145921,
        max_contributors=461,final_valid_exceed192=334072,final_valid_exceed192_fraction=s['valid_depth_counts']['capacity_exceed']['192']['fraction'],
        source_and_full_depth_slope_audit_pass=True,runtime_all_original_outputs_bitwise=True,count_mismatches=0,
        counting_overhead_paired_ratio=s['paired_count_instrumentation_ratio_geometric_mean'],
        terminal_handles=dict(full24_session11252=0,first_verifier_encoding_failure_exit_code=1,utf8_corrected_verifier_exit_code=0,snapshot_session9591=0),
        source_latex_matches_builtin_success=True,source_canvas_matches_syntax_checked_stage=True,
        canvas_syntax_checker_rows24_erratum='Historical constant, not tab or page count; SDK/runtime not verified.',
        snapshot=dict(sha256=manifest['archive_sha256'],files=len(manifest['files_sha256']),bytes=snapshot.stat().st_size,selected_member_checks=names),
        limits=['Counts in native affine renderer only, not a certified complete-image depth implementation or source-parameter backward/GT result.',
                '512 covers only this observed cohort; dynamic lists must preserve future inputs without truncation.',
                'Runtime RGB/alpha/normal/radii compared in full; only median/counters retained for independent offline pixel audit.'],
        next='Capture actual first-iteration IDs in dynamic CSR lists, verify complete pixel counts/order and output identity, then scalable parameter cache/parallel certificates and explicit unresolved/world-VJP gates.')
    out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
