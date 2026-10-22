"""Seal packed pipeline, measured tradeoffs and stronger-baseline/telemetry limits."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_packed_query_v25'

def main():
    assert not (OUT/'evidence_manifest.json').exists() and not list(OUT.rglob('*failure*'))
    r=json.loads((OUT/'results.json').read_text());audit=json.loads((OUT/'independent_audit.json').read_text());a=json.loads((OUT/'timing/independent_analysis.json').read_text())
    assert r['all_passed'] and r['new_native_forwards']==8 and len(r['rows'])==32 and audit['all_passed'] and len(audit['rows'])==32
    assert json.loads((OUT/'fixtures.json').read_text())['all_passed'] and all(x['patch_literal'] and x['scores_literal_v24'] for x in r['rows'])
    assert a['all_verified'] and a['native_forwards']==1004 and a['blocks']==32 and a['measured_endpoints']==128
    assert len(a['rows'])==4 and all(x['methods']['D_batch_packed']['peak_increment_bytes']>x['methods']['C_sequential_packed']['peak_increment_bytes'] for x in a['rows'])
    dep=set()
    for name in ['plan.json','runtime_dependency_addendum.json','timing/plan.json','timing/independent_analysis.json','timing/telemetry_audit_addendum.json','figure_provenance.json']:
        p=json.loads((OUT/name).read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        dep.update(p['inputs'])
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'] and sha(ROOT/'tgs_packed_patch_v25.cu')==b['source_sha256']
    dep.update(['TGS_PACKED_QUERY_RESULT_2026_10_21.md','seal_tgs_packed_query_v25.py','audit_tgs_packed_query_v25.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,RGB_and_scores_literal_cases=32,
        correctness_native_forwards=8,timing_native_forwards=1004,phase_total_native_forwards=1012,training=False,backward_exercised=False,accepted_mutations=0,
        same_FP64_oracle_timing=True,shared_baseline_moment_strengthening_still_required=True,clock_telemetry_anomaly_preserved=True,
        independent_scene_test=False,quality_or_policy_gain_established=False,novelty_established=False,goal_complete=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
