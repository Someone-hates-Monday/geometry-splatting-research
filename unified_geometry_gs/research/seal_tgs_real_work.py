"""Seal v7 including original failure, corrected sentinel, all raw distributions and notes."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_real_work_v7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=OUT/'evidence_manifest.json';assert not manifest.exists()
    plan=json.loads((OUT/'input_plan.json').read_text());initial=json.loads((OUT/'results.json').read_text());fix=json.loads((OUT/'sentinel_v7_1/results.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text());theory=json.loads((OUT/'active_interval_checks.json').read_text());fig=json.loads((OUT/'figure_metadata.json').read_text())
    assert not initial['all_gates'] and fix['all_gates'] and audit['all_gates'];assert audit['repair_results_sha256']==sha(OUT/'sentinel_v7_1/results.json')
    assert theory['audit_sha256']==fig['audit_sha256']==sha(OUT/'independent_audit.json');assert theory['note_sha256']==sha(ROOT/'TGS_ACTIVE_EDGE_INTERVAL_NOTE.md');assert theory['script_sha256']==sha(ROOT/'check_tgs_active_edge_interval.py');assert audit['script_sha256']==sha(ROOT/'audit_tgs_real_work.py');assert fig['script_sha256']==sha(ROOT/'plot_tgs_real_work.py')
    assert plan['protocol_sha256']==sha(ROOT/'TGS_REAL_WORK_PROTOCOL_V7.md');assert plan['source_sha256']==sha(ROOT/'tgs_real_work_probe.cu');assert plan['prepare_sha256']==sha(ROOT/'prepare_tgs_real_work.py')
    for name,h in fig['figure_files'].items():assert sha(OUT/name)==h
    checked=0
    for v,k in [('tgs_phase_cost_v5','v5_manifest_sha256'),('tgs_compensated_edge_v6','v6_manifest_sha256')]:
        p=ROOT/'experiments'/v/'evidence_manifest.json';assert sha(p)==plan[k]
        for e in json.loads(p.read_text())['files']:assert sha(ROOT/e['path'])==e['sha256'];checked+=1
    decision=dict(observer_endpoint_gate=True,initial_endpoint_gate=False,correction_only_declared_sentinel=True,actual_work_profile_gate=True,cache_and_sample_audit=True,active_interval_lemma_checked=True,active_interval_implementation=False,conditional_power_allowance_only=True,power_interval_proof=False,renderer_candidate_gate=False,training=False,timing=False,performance_promotion=False,production_promotion=False,visual_inspection='K_survival.png inspected: readable five curves, log axis, capacity mark, limitations and legend; no clipped content')
    (OUT/'promotion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    names=['TGS_REAL_WORK_PROTOCOL_V7.md','TGS_REAL_WORK_ADDENDUM_V7_1.md','TGS_REAL_WORK_RESULT_2026_10_20.md','TGS_ACTIVE_EDGE_INTERVAL_NOTE.md','tgs_real_work_probe.cu','prepare_tgs_real_work.py','profile_tgs_real_work.py','repair_tgs_real_work_sentinel.py','audit_tgs_real_work.py','check_tgs_active_edge_interval.py','plot_tgs_real_work.py','seal_tgs_real_work.py']
    files=[ROOT/p for p in names]+[p for p in OUT.rglob('*') if p.is_file()]
    entries=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    manifest.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=entries,prior_manifest_files_checked=checked,v5_manifest_sha256=plan['v5_manifest_sha256'],v6_manifest_sha256=plan['v6_manifest_sha256']),indent=2)+'\n')
    print(json.dumps(dict(files=len(entries),bytes=sum(e['size'] for e in entries),manifest_sha256=sha(manifest),prior_hash_checks=checked,goal_active=True)),flush=True)
if __name__=='__main__':main()
