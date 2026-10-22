"""Seal coupled-certificate evidence without inferring an academic/method success."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_coupled_v18'

def main():
    d=json.loads((OUT/'development_audit.json').read_text());f=json.loads((OUT/'fresh/independent_audit.json').read_text());s=json.loads((OUT/'structure_audit.json').read_text())
    assert d['all_passed'] and f['all_passed'] and s['all_passed']
    assert d['aggregate']['exact_zero_certificates']==51 and f['aggregate']['exact_zero_certificates']==52
    assert s['local_channel_checks']==768 and s['literal_zero_pairs']==103
    result=json.loads((OUT/'fresh/results.json').read_text());assert result['full_forwards']==6 and result['trace_calls']==256
    dependency=set()
    for name in ['development_plan.json','fresh/plan.json','structure_plan.json']:
        plan=json.loads((OUT/name).read_text())
        for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
        dependency.update(plan['inputs'])
    extra=['TGS_COUPLED_RESULT_2026_10_21.md','audit_tgs_coupled_fresh_v18.py','audit_tgs_coupled_structure_v18.py',
        'analyze_tgs_coupled_limits_v18.py','seal_tgs_coupled_v18.py']
    dependency.update(extra)
    files=sorted(p for p in OUT.rglob('*') if p.is_file());assert not (OUT/'evidence_manifest.json').exists()
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    deps=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in sorted(dependency)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=deps,training=False,goal_complete=False,
        conditional_fixed_input_bounds_verified=True,whole_image_split_rule_validated=False,measured_speedup=False,
        GT_improvement=False,novelty_established=False))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(deps),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
