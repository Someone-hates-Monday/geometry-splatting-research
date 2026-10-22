"""Freeze formula before posthoc application to every v17 pair."""
import json
from pathlib import Path
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_coupled_certificate_v18 import coupled,aggregate
OLD=ROOT/'experiments/tgs_interleaving_v17/validation';OUT=ROOT/'experiments/tgs_coupled_v18'

def main():
    OUT.mkdir(exist_ok=True);p=json.loads((OLD/'plan.json').read_text());r=json.loads((OLD/'results.json').read_text())
    assert json.loads((OLD/'independent_audit_v2.json').read_text())['all_passed']
    files=[ROOT/'TGS_COUPLED_CERTIFICATE_NOTE_V18.md',ROOT/'tgs_coupled_certificate_v18.py',Path(__file__),
        ROOT/'tgs_interleaving_certificate_v17.py',ROOT/'audit_tgs_interleaving_validation_v17_v2.py',
        ROOT/'audit_tgs_compensated_depth_trace.py',OLD/'results.json',OLD/'independent_audit_v2.json',OLD.parent/'evidence_manifest.json']
    for row in r['rows']:files.extend([OLD/row['before'],OLD/row['after']])
    write_new(OUT/'development_plan.json',dict(inputs={str(x.relative_to(ROOT)):sha(x) for x in files},
        development_only=True,formula_before_new_outputs=True,training=False))
    rows=[]
    for row in r['rows']:
        rows.append(dict(scene=row['scene'],camera=row['camera'],event=row['event'],pixel=row['pixel'],
            **coupled(OLD/row['before'],OLD/row['after'],row['parent_id'],row['N'])))
    out=dict(all_passed=True,aggregate=aggregate(rows),rows=rows,posthoc_development=True,training=False)
    write_new(OUT/'development_audit.json',out);print(json.dumps(out['aggregate']),flush=True)

if __name__=='__main__':main()
