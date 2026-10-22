"""Evidence-bounded development summary with the live pilot status as a snapshot."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output required.')
    sources={}
    def read(name):
        p=ROOT/'research'/name
        sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    comparisons=[]
    for name,label in (('solver_backward_repeat_v1.json','four'),('iqi_backward_v1.json','iqi4'),
        ('adjoint_backward_v1.json','adj4'),('adjquad_backward_v1.json','adjquad4')):
        data=read('results/'+name)
        values=[c for row in data['rows'] for c in row['gradient_comparison'].values()]
        comparisons.append({'variant':label,'groups':len(values),
            'passed_groups_at_relative_l2_0_001':sum(c['relative_l2_drift']<=.001 for c in values),
            'max_relative_l2_drift':max(c['relative_l2_drift'] for c in values),
            'max_control_repeat_relative_l2_drift':max(c['control_repeat_relative_l2_drift'] for c in values),
            'all_finite':all(c['all_finite'] for c in values)})
    timing=read('results/adjquad_step_timing_v1.json')
    selected=[r for r in timing['rows'] if r['variant']=='adjquad4']
    ratios=[r['latency_ratio_to_control'] for r in selected]
    pilot=read('experiments/adjquad55_v1/manifest.json')
    smoke=read('experiments/adjquad_smoke55_v1/manifest.json')
    restore=read('results/checkpoint_restore_v1.json')
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'gradient_comparison':comparisons,
        'frozen_rgb_plus_depth_forward_backward_timing':{
            'views':len(selected),'rounds':timing['rounds'],'repeats':timing['repeats'],
            'latency_reduction_percent_range':[100*(1-max(ratios)),100*(1-min(ratios))],
            'median_view_latency_ratio_to_control':statistics.median(ratios),
            'limitations':timing['limitations']},
        'checkpoint_restore':restore['cases'],
        'smoke_branch_state_verified':smoke['branch_state_verified'],
        'smoke_initial_state_sha256':smoke['initial_state_sha256'],
        'full_pilot_status_snapshot':{'status':pilot['status'],'stage':pilot['steps'][-1]['stage'],
            'branch_state_verified':pilot['branch_state_verified'],
            'warning':'Final branch_state_verified is only written after both arms finish; a running snapshot is not a failure.'},
        'source_sha256':sources,
        'interpretation':'Promising development engineering result; full training quality and academic novelty remain unestablished.'}
    sources['summarize_adjoint_development.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'gradient_comparison':comparisons,'timing':result['frozen_rgb_plus_depth_forward_backward_timing'],
        'pilot':result['full_pilot_status_snapshot']},indent=2))


if __name__=='__main__':main()
