"""Seal joint policy negative results and explicitly scoped mathematical supplements."""
import json
from pathlib import Path
from tgs_joint_policy_data_v29 import ROOT,OUT,SOURCE,sha,write
def main():
    assert not (OUT/'evidence_manifest.json').exists() and not list(OUT.rglob('*failure*'))
    a=json.loads((OUT/'analysis.json').read_text());au=json.loads((OUT/'independent_audit.json').read_text());reg=json.loads((OUT/'global_reg_results.json').read_text());obs=json.loads((OUT/'SH_observability_results.json').read_text());choice=json.loads((OUT/'choices.json').read_text())
    assert a['all_verified'] and au['all_passed'] and a['native_forwards']==52 and a['joint_references']==40 and reg['all_identities_passed']
    assert all(s['all_design_ranks_two'] and len(s['rows'])==40 for s in obs['scenes']) and choice['calibration_only'] and not choice['heldout_payloads_seen']
    assert json.loads((OUT/'heldout_identity.json').read_text())['all_passed']
    assert all(s['policies'][0]['calibration']['mean_mixed']<0 and s['policies'][0]['heldout']['mean_mixed']>0 for s in a['scenes'])
    dep=set()
    for folder in [OUT,SOURCE]:
        for path in folder.rglob('*.json'):
            x=json.loads(path.read_text())
            if isinstance(x,dict) and isinstance(x.get('inputs'),dict):
                for n,h in x['inputs'].items():assert sha(ROOT/n)==h,(path,n);dep.add(n)
    for f in json.loads((SOURCE/'manifest.json').read_text())['files']:assert sha(ROOT/f['local'])==f['sha256'] and f['zip_crc_checked'];dep.add(f['local'])
    for n in ['TGS_JOINT_POLICY_RESULT_2026_10_21.md','TGS_RESEARCH_POSITION_UPDATE_V29.md','TGS_BOUNDED_APPEARANCE_NEXT_PLAN_V29.md','seal_tgs_joint_policy_v29.py']:dep.add(n)
    entries=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for folder in [OUT,SOURCE] for p in sorted(folder.rglob('*')) if p.is_file()]
    deps=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write(OUT/'evidence_manifest.json',dict(files=entries,dependencies=deps,native_forwards=52,full_joint_references=40,candidates_per_scene=256,edits_per_forced_policy=8,new_heldout_views=4,calibration_mosaics_literal=True,
        primary_heldout_mean_image_form_worse_both_scenes=True,fixed_budget_reg_real_input_identities_verified=True,SH_rank_diagnostic_not_fitting=True,finite_precision_contract='fixed_external_FP32_colors_child_only_SH',
        training=False,backward=False,accepted_mutations=0,full_training_objective_measured=False,geometry_measured=False,novelty_or_final_quality_gain_established=False,goal_complete=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(f['bytes'] for f in entries),dependencies=len(deps),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)
if __name__=='__main__':main()
