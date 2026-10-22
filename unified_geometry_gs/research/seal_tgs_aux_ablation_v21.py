"""Seal selected-output matched control and preserve mixed setup-inclusive results."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_aux_ablation_common_v21 import OUT

def main():
    v=json.loads((OUT/'validation/results.json').read_text());a=json.loads((OUT/'timing/independent_analysis.json').read_text())
    assert v['all_gates_passed'] and v['full_forwards']==204 and len(v['rows'])==48 and len(v['baselines'])==6
    assert a['all_verified'] and a['blocks']==128 and a['measured_endpoints']==512
    assert next(x for x in a['views'] if x['scene']=='flowers')['estimated_setup_inclusive_ratio']<1
    dep=set()
    for name in ['plan.json','timing/plan.json']:
        p=json.loads((OUT/name).read_text())
        for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
        dep.update(p['inputs'])
    p=json.loads((OUT/'plan.json').read_text());assert sha(Path(p['native']['path']))==p['native']['sha256']
    b=json.loads((OUT/'build_receipt.json').read_text());assert sha(Path(b['path']))==b['sha256'] and sha(OUT/'source/forward.cu')==b['generated_forward_sha256']
    validation_plan=json.loads((OUT/'validation/plan.json').read_text());assert sha(ROOT/'validate_tgs_aux_ablation_v21.py')==validation_plan['script_sha256']
    dep.update(['validate_tgs_aux_ablation_v21.py','analyze_tgs_aux_ablation_v21.py','plot_tgs_aux_ablation_v21.py','TGS_AUX_ABLATION_RESULT_2026_10_21.md','seal_tgs_aux_ablation_v21.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());assert not (OUT/'evidence_manifest.json').exists()
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files];dependencies=[dict(path=n,bytes=(ROOT/n).stat().st_size,sha256=sha(ROOT/n)) for n in sorted(dep)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=dependencies,goal_complete=False,training=False,backward_exercised=False,
        selected_outputs_literal_on48_cases=True,aux_arithmetic_matched_timing=True,API_allocations_preserved=True,
        flowers_setup_inclusive_estimate_slower=True,novelty_established=False,model_quality_gain=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(dependencies),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
