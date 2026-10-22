"""Seal the corrected-contract transfer pilot including every preserved failure."""
import json
from pathlib import Path
from prepare_tgs_policy_data_v28 import ROOT,OUT,SOURCE,sha,write
def main():
    assert not (OUT/'evidence_manifest.json').exists();a=json.loads((OUT/'analysis.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text());assert a['all_verified'] and audit['all_passed']
    assert a['total_native_forwards']==56 and a['selected_full_references']==36 and audit['all_selected_references_verified']
    choices=json.loads((OUT/'choices.json').read_text());assert choices['calibration_only'] and not choices['test_scores_seen'] and not choices['heldout_payloads_seen']
    assert json.loads((OUT/'heldout_identity.json').read_text())['all_passed']
    failures=sorted(str(p.relative_to(OUT)) for p in OUT.rglob('*failure.txt'))
    assert failures==['fetch_heldout_failure.txt','heldout_fetch_recovery_failure.txt','reference_calibration_failure.txt'],failures
    dep=set()
    for folder in [OUT,SOURCE]:
        for path in folder.rglob('*.json'):
            data=json.loads(path.read_text())
            if isinstance(data,dict) and isinstance(data.get('inputs'),dict):
                for n,h in data['inputs'].items():assert sha(ROOT/n)==h,(path,n);dep.add(n)
    for cohort in ['calibration','heldout']:
        manifest=json.loads((SOURCE/(cohort+'_manifest.json')).read_text())
        for row in manifest['files']:assert sha(ROOT/row['local'])==row['sha256'] and row['zip_crc_checked'];dep.add(row['local'])
    for n in ['TGS_POLICY_TRANSFER_RESULT_2026_10_21.md','TGS_POLICY_TRANSFER_PROTOCOL_V28.md','TGS_POLICY_REFERENCE_ADDENDUM_V28_R1.md','run_tgs_policy_transfer_v28.py','reference_tgs_policy_v28_r1.py','diagnose_tgs_policy_reference_v28.py','recover_tgs_policy_heldout_fetch_v28.py','recover_tgs_policy_heldout_windows_v28.py','audit_tgs_policy_transfer_v28.py','analyze_tgs_policy_transfer_v28.py','plot_tgs_policy_transfer_v28.py','seal_tgs_policy_transfer_v28.py']:dep.add(n)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for folder in [OUT,SOURCE] for p in sorted(folder.rglob('*')) if p.is_file()]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,total_native_forwards=56,selected_references=36,candidates_per_scene=32,scenes=2,calibration_views=4,heldout_views=4,choices_frozen_before_test_payloads=True,
        corrected_numerical_contract='fixed_external_FP32_colors_child_only_SH',original_full_recompute_reference_failed=True,whole_model_SH_recompute_equivalence=False,
        three_original_failure_records_preserved=True,windows_TLS_recovery=True,unselected_scores_not_all_full_native_audited=True,training=False,backward=False,accepted_mutations=0,final_trained_quality_gain=False,geometry_quality_measured=False,novelty_established=False,goal_complete=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)
if __name__=='__main__':main()
