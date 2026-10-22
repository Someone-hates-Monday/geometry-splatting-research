"""Posthoc limitation probe; no formula changes or new data selection claim."""
import json
from fractions import Fraction as F
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
from tgs_interleaving_certificate_v17 import q,layers,certificate
OUT=ROOT/'experiments/tgs_coupled_v18'

def main():
    audit=json.loads((OUT/'fresh/independent_audit.json').read_text());r=max((x for x in audit['rows'] if x['actual_peak']>0),key=lambda x:x['coupled']['ratio'])
    a=np.load(OUT/'fresh'/r['before']);b=np.load(OUT/'fresh'/r['after']);old=layers(a['packed'],a['full']);new=layers(b['packed'],b['full'])
    if not any(x[0]==r['parent_id'] for x in old):old=[(r['parent_id'],F(0),[q(v) for v in a['parent_color']])]+old
    c=certificate(old,new,r['parent_id'],r['N'])
    case=dict(scene=r['scene'],event=r['event'],pixel=r['pixel'],actual_peak=r['actual_peak'],coupled_bound=r['coupled']['bound_peak'],
        coupled_ratio=r['coupled']['ratio'],active_crossed_external=c['active_crossed_external'],
        real_signed_window=[float(x) for x in c['window']],real_signed_order=[float(x) for x in c['order']],
        real_full_difference=[float(x) for x in c['actual']],window_abs_bound=[float(x) for x in c['window_bound']],order_abs_bound=[float(x) for x in c['order_bound']])
    cohorts={n:json.loads(p.read_text())['aggregate'] for n,p in [('development',OUT/'development_audit.json'),('fresh',OUT/'fresh/independent_audit.json')]}
    counts={n:dict(full_candidate_records=x['candidate_records'],accepted_records=x['accepted_records'],divergent_suffix_records=x['divergent_suffix_records'],
        suffix_over_accepted=x['divergent_suffix_records']/x['accepted_records']) for n,x in cohorts.items()}
    write_new(OUT/'limitations_audit.json',dict(posthoc=True,worst_ratio_probe=case,record_counts=counts,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'fresh/independent_audit.json',OUT/'development_audit.json',ROOT/'analyze_tgs_coupled_limits_v18.py']},
        exact_signed_counterfactual_alpha_information_still_required=True,measured_speedup=False,
        decision='Retain as conditional analysis/reference; do not promote to a practical adaptive-splitting rule'))
    print(json.dumps(dict(worst=case,counts=counts)),flush=True)

if __name__=='__main__':main()
