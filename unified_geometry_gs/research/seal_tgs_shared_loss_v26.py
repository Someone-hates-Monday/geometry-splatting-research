"""Seal shared-baseline fairness, exact identities and charged setup costs."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_shared_loss_v26';V25=ROOT/'experiments/tgs_packed_query_v25'

def main():
    assert not (OUT/'evidence_manifest.json').exists() and not list(OUT.rglob('*failure*'))
    r=json.loads((OUT/'results.json').read_text());a=json.loads((OUT/'receipt_audit.json').read_text());t=json.loads((OUT/'timing/independent_analysis.json').read_text())
    assert r['all_passed'] and r['new_native_forwards']==0 and len(r['rows'])==32 and a['all_verified'] and a['real_cases']==32
    assert a['cached_unshared_scores_literal'] and a['cached_unshared_maps_runtime_gate_passed'] and a['max_CPU_score_error']<1e-13
    assert t['all_verified'] and t['native_forwards']==1004 and t['blocks']==32 and t['measured_endpoints']==128
    assert json.loads((OUT/'fixtures.json').read_text())['all_passed']
    dep=set()
    for path in [OUT/'plan.json',OUT/'receipt_audit.json',OUT/'timing/plan.json',OUT/'timing/independent_analysis.json',OUT/'figure_provenance.json',V25/'plan.json',V25/'runtime_dependency_addendum.json']:
        p=json.loads(path.read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        dep.update(p['inputs'])
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'] and sha(ROOT/'tgs_shared_image_loss_v26.cu')==b['source_sha256']
    dep.update(['TGS_SHARED_LOSS_RESULT_2026_10_21.md','TGS_RESEARCH_POSITION_UPDATE_V26.md','seal_tgs_shared_loss_v26.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,real_cases=32,shared_unshared_GPU_scores_and_maps_literal=True,
        same_shared_cache_in_all_four_timing_arms=True,loss_setup_charged_to_both_when_needed=True,new_native_forwards=1004,
        training=False,backward_exercised=False,accepted_mutations=0,frequency_stability_established=False,
        independent_scene_or_final_quality_gain=False,novelty_established=False,goal_complete=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
