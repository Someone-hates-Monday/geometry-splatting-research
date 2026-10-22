"""Verify terminal shared-cache stage, editor source and frozen snapshot."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=R/'results/shared_cache_checkpoint_v1.json';assert not out.exists()
    p=R/'snapshots/after_shared_cache_v1.zip';m=read(p.with_suffix('.manifest.json'));assert sha(p)==m['archive_sha256']
    c=read(R/'results/latex_compile_status_v32.json');assert c['kind']=='success' and sha(R/'median_depth_note.tex')==c['source_sha256']
    g=read(R/'results/canvas_shared_cache_guard_v1.json');assert sha(Path(g['destination']))==g['staged_sha256'] and not read(R/'results/canvas_shared_cache_syntax_v1.json')['errors']
    s=read(R/'experiments/shared_cache_v1/summary.json');a=read(R/'results/shared_cache_audit_v1.json')
    assert s['fixed']['all_bitwise_pass'] and s['fixed']['local_reuse_gate_pass'] and s['all24_real_builds'] and s['all_real_cache_constructible']
    assert a['all_saved_budget_and_root_bits_match'] and a['frozen_core_and_query_body_unchanged']
    names=['research/shared_cache_cuda.py','research/benchmark_shared_cache.py','research/verify_shared_cache.py','research/SHARED_CACHE_PROTOCOL.md',
           'research/SHARED_CACHE_RESULT.md','research/experiments/shared_cache_v1/summary.json','research/experiments/shared_cache_v1/fixed_outputs.npz',
           'research/results/shared_cache_audit_v1.json','research/median_depth_note.tex','research/results/latex_compile_status_v32.json','research/canvas_result_snapshot.tsx']
    with zipfile.ZipFile(p) as z:
        assert json.loads(z.read('SNAPSHOT_MANIFEST.json'))['files_sha256']==m['files_sha256']
        for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==m['files_sha256'][n]
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,
        previous_goal_turn_classification='progress: dynamic actual CSR all24/all870178922 checked including failure guards',
        current_goal_turn_classification='progress: query-independent shared cache preserves all fixed budgets/roots; local repeat timing gate and all24 real primitive builds completed',
        source_rows=7322,repeated_queries=58576,all_fixed_budget_root_bits_match=True,local_repeat_paired_ratio=s['fixed']['paired_ratio_geometric_mean'],
        real_views=24,all_real_builds_finite_status0=True,cache_row_bytes=224,real_build_event_ms_range=[min(r['median_event_ms'] for r in s['rows']),max(r['median_event_ms'] for r in s['rows'])],
        terminal_handles=dict(shared_cache_session55325=0,independent_CPU_audit_exit_code=0,snapshot_session2078=0),
        source_latex_matches_builtin_success=True,source_canvas_matches_syntax_checked_stage=True,
        canvas_syntax_checker_rows24_erratum='Historical constant, not tab/row count; SDK/runtime not verified.',
        snapshot=dict(sha256=m['archive_sha256'],files=len(m['files_sha256']),bytes=p.stat().st_size,selected_member_checks=names),
        limits=['Diagnostic224B cache plus288B/source rows, not the proposed56B production layout.',
                'Real cache construction only; all870M actual record decodes and full roots/VJP/GT/training not yet verified.',
                'Real cache content hashes/runtime statistics recorded; arrays not retained for second offline/MP full audit.'],
        next='Establish actual pixel-query/AMP sources and batch CSR cache decode without full diagnostic arrays resident; then dynamic/ray-parallel roots preserving original target/range and explicit unresolved/F32 output rounding.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
