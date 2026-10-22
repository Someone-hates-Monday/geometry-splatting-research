"""Audit terminal actual CSR decode stage and recoverable source/results snapshot."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,zipfile
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    out=R/'results/csr_decode_checkpoint_v1.json';assert not out.exists()
    p=R/'snapshots/after_csr_decode_v1.zip';m=read(p.with_suffix('.manifest.json'));assert sha(p)==m['archive_sha256']
    c=read(R/'results/latex_compile_status_v33.json');assert c['kind']=='success' and sha(R/'median_depth_note.tex')==c['source_sha256']
    g=read(R/'results/canvas_csr_decode_guard_v1.json');assert sha(Path(g['destination']))==g['staged_sha256'] and not read(R/'results/canvas_csr_decode_syntax_v1.json')['errors']
    s=read(R/'experiments/csr_decode_v1/summary.json');a=read(R/'results/csr_decode_audit_v1.json')
    assert s['all24_views'] and s['complete_record_count_pass'] and s['all_high_path_valid'] and s['all_fixed_probes_bitwise'] and a['gate_pass']
    names=['research/csr_decode_cuda.py','research/benchmark_csr_decode.py','research/verify_csr_decode.py','research/CSR_DECODE_PROTOCOL.md',
           'research/CSR_DECODE_RESULT.md','research/experiments/csr_decode_v1/summary.json','research/experiments/csr_decode_v1/scan55_0000_probes.npz',
           'research/experiments/csr_decode_v1/scan55_0000_query_amp.npz','research/results/csr_decode_audit_v1.json',
           'research/median_depth_note.tex','research/results/latex_compile_status_v33.json','research/canvas_result_snapshot.tsx']
    with zipfile.ZipFile(p) as z:
        assert json.loads(z.read('SNAPSHOT_MANIFEST.json'))['files_sha256']==m['files_sha256']
        for n in names:assert hashlib.sha256(z.read(n)).hexdigest()==m['files_sha256'][n]
    result=dict(status='current_step_complete',checked_utc=datetime.now(timezone.utc).isoformat(),overall_goal_achieved=False,
        previous_goal_turn_classification='progress: shared cache preserves fixed output bits and all24 real primitive caches built',
        current_goal_turn_classification='progress: all870178922 actual-query/AMP CSR records decoded with valid finite states;1924 fixed MP probes pass; excessive diagnostic cost measured',
        views=24,records=870178922,all_two_mode_states0_and_valid_finite=True,world_probes=1924,MP80_140_coefficient_checks=15392,
        old_anomaly_query_mismatches=0,nominal_double_alpha_underflow_zero_per_mode=[sum(r['nominal_alpha_underflow_zero'][mode] for r in s['rows']) for mode in [0,1]],
        decode_event_total_ms_per_view_range=[min(r['decode_event_total_ms'] for r in s['rows']),max(r['decode_event_total_ms'] for r in s['rows'])],
        max_temporary_output_bytes=27262976,terminal_handles=dict(csr_decode_session35967=0,independent_MP_verifier_exit_code=0,snapshot_session73978=0),
        source_latex_matches_builtin_success=True,source_canvas_matches_syntax_checked_stage=True,
        canvas_syntax_checker_rows24_erratum='Historical constant, not tab/row count; SDK/runtime not verified.',
        snapshot=dict(sha256=m['archive_sha256'],files=len(m['files_sha256']),bytes=p.stat().st_size,selected_member_checks=names),
        limits=['Full actual record state/accounting, not individual MP checks of all870M or image root certificates.',
                'Full double streams not retained; output hash manifests cannot be completely recomputed offline.',
                'Both-mode26double diagnostics take0.60–2.16sec/view before CPU transfer; not adopted as method/training speed.'],
        next='Cheap sufficient endpoint certificates on actual inputs, selective strict fallback, compact outputs/ray-parallel roots, explicit unresolved and publicF32 rounding, then sourceVJP/fullGT/training gates.')
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
