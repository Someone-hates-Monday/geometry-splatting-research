"""Freeze evidence for conditional mathematics; no scientific-success inference."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_interleaving_v17'

def main():
    dev=json.loads((OUT/'development_audit.json').read_text());audit=json.loads((OUT/'validation/independent_audit_v2.json').read_text())
    assert dev['all_passed'] and dev['symbolic_interleavings']==18 and audit['all_passed']
    assert audit['aggregate']['pixels']==64 and audit['aggregate']['early_stop_pairs']==20
    assert audit['aggregate']['nonzero_error']==13 and audit['aggregate']['zero_error']==51
    dependency=set()
    for name in ['development_plan.json','validation/plan.json']:
        plan=json.loads((OUT/name).read_text())
        for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
        dependency.update(plan['inputs'])
    original=json.loads((OUT/'validation/audit_plan.json').read_text());fixed=json.loads((OUT/'validation/audit_plan_v2.json').read_text())
    assert sha(ROOT/'audit_tgs_interleaving_validation_v17.py')==original['script_sha256']
    assert sha(ROOT/'audit_tgs_interleaving_validation_v17_v2.py')==fixed['script_sha256']
    assert sha(ROOT/'audit_tgs_compensated_depth_trace.py')==fixed['RN_helper_sha256']
    dependency.update(['TGS_INTERLEAVING_RESULT_2026_10_21.md','audit_tgs_interleaving_validation_v17.py',
        'audit_tgs_interleaving_validation_v17_v2.py','audit_tgs_compensated_depth_trace.py','plot_tgs_interleaving_v17.py',
        'seal_tgs_interleaving_v17.py'])
    files=sorted(p for p in OUT.rglob('*') if p.is_file());assert not (OUT/'evidence_manifest.json').exists()
    entries=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    deps=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(ROOT/p)) for p in sorted(dependency)]
    write_new(OUT/'evidence_manifest.json',dict(files=entries,dependencies=deps,training=False,goal_complete=False,
        native_error_contained_on_frozen_pixels=True,practical_split_rule_validated=False,novelty_established=False,
        scope='Conditional fixed-cache per-pixel certificate; known views and directed pixel selection'))
    print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),dependencies=len(deps),sha256=sha(OUT/'evidence_manifest.json'))),flush=True)

if __name__=='__main__':main()
