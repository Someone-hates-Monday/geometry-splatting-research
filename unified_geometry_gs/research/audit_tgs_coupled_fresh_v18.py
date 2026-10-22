"""Unchanged coupled formula on separate event/camera inputs; all points retained."""
import json
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new,exact
from tgs_coupled_certificate_v18 import coupled,aggregate
OUT=ROOT/'experiments/tgs_coupled_v18/fresh'

def main():
    plan=json.loads((OUT/'plan.json').read_text());results=json.loads((OUT/'results.json').read_text())
    assert results['full_forwards']==6 and results['trace_calls']==256 and results['all_native_endpoints_literal']
    assert results['plan_sha256']==sha(OUT/'plan.json')
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    write_new(OUT/'audit_plan.json',dict(script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'results.json'),training=False))
    rows=[]
    for row in results['rows']:
        got=coupled(OUT/row['before'],OUT/row['after'],row['parent_id'],row['N'])
        if got['exact_zero_by_input_equality']:
            a=np.load(OUT/row['before'])['summary'][:3];b=np.load(OUT/row['after'])['summary'][:3];assert exact(a,b)
        rows.append(dict(**row,**got))
    out=dict(all_passed=True,aggregate=aggregate(rows),rows=rows,formula_unchanged=True,training=False,
        general_novelty_established=False,full_image_split_rule_validated=False,GT_improvement=False)
    write_new(OUT/'independent_audit.json',out);print(json.dumps(out['aggregate']),flush=True)

if __name__=='__main__':main()
