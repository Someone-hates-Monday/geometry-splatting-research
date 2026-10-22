"""Seal the aggregate-null joint study and explicitly scoped regularizer correction."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_joint_edit_common_v27 import OUT

def main():
    assert not (OUT/'evidence_manifest.json').exists() and not list(OUT.rglob('*failure*'))
    r=json.loads((OUT/'results.json').read_text());a=json.loads((OUT/'independent_audit.json').read_text());summary=json.loads((OUT/'analysis.json').read_text())
    assert r['all_passed'] and len(r['rows'])==24 and r['new_native_forwards']==32 and a['all_passed'] and summary['all_verified']
    assert a['point_disjoint_pairs']==a['ssim_disjoint_pairs']==22 and a['nontrivial_ssim_disjoint_pairs']==19
    assert json.loads((OUT/'fixtures.json').read_text())['all_passed'] and json.loads((OUT/'global_reg_results.json').read_text())['all_identities_passed']
    for row in r['rows']:
        assert row['external_child_visible_fields_literal'] and row['RGB_and_SSIM_unchanged_outside_structural_union']
        if row['classification']['point_disjoint']:assert row['point_RGB_mosaic_literal'] and row['PNG8_interaction']==0
        if row['classification']['ssim_disjoint']:assert row['SSIM_mosaic_literal']
        assert all(abs(row['interactions'][k])<=row['roundoff_bounds'][k] for k in row['required_zero_metric_indices'])
    dep=set()
    for name in ['plan.json','independent_audit.json','support_diagnostic_plan.json','global_reg_plan.json','analysis.json']:
        p=json.loads((OUT/name).read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        dep.update(p['inputs'])
    assert sha(OUT/'support_diagnostic_plan.json')==json.loads((OUT/'support_diagnostic_results.json').read_text())['plan_sha256']
    assert sha(OUT/'global_reg_plan.json')==json.loads((OUT/'global_reg_results.json').read_text())['plan_sha256']
    dep.update(['TGS_JOINT_EDIT_RESULT_2026_10_21.md','seal_tgs_joint_edit_v27.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,new_native_forwards=32,joint_real_cases=24,
        structural_disjoint_pairs=22,nontrivial_disjoint_pairs=19,real_image_interaction_beyond_roundoff_found=False,
        post_hoc_support_diagnostic_explicit=True,global_regularizer_symbolic_not_measured=True,
        training=False,backward_exercised=False,accepted_mutations=0,independent_scene_or_final_quality_gain=False,novelty_established=False,goal_complete=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
