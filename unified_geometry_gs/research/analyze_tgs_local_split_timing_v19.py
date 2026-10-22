"""Audit balanced paired timing data; descriptive clustered-by-case summaries only."""
import json,math
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_local_split_v19/timing'

def median(x):return float(np.median(x))

def main():
    p=json.loads((OUT/'plan.json').read_text());r=json.loads((OUT/'results.json').read_text());assert r['full_forwards']==644 and len(r['rows'])==32
    for name,h in p['inputs'].items():assert sha(ROOT/name)==h,name
    summaries=[]
    for row in r['rows']:
        key=f"{row['scene']}_c{row['camera']}_e{row['event']}";pairs=row['pairs'];assert len(pairs)==8 and [x['repeat'] for x in pairs]==list(range(8))
        assert [x['order'] for x in pairs]==['full_then_local' if x==0 else 'local_then_full' for x in p['orders'][key]]
        assert sum(x['order']=='full_then_local' for x in pairs)==4
        for pair in pairs:
            for name in ['full','local']:
                for field in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']:assert math.isfinite(pair[name][field]) and pair[name][field]>0
            assert pair['full_over_local_wall']==pair['full']['wall_ms']/pair['local']['wall_ms']
        out=dict(scene=row['scene'],camera=row['camera'],event=row['event'],affected_tiles=row['affected_tiles'],
            affected_pixels=row['affected_pixels'],pixels=row['pixels'],paired_wall_ratio=median([x['full_over_local_wall'] for x in pairs]))
        for method in ['full','local']:
            out[method]={field:median([x[method][field] for x in pairs]) for field in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']}
        out['incremental_workspace_ratio']=out['full']['peak_increment_bytes']/out['local']['peak_increment_bytes'];summaries.append(out)
    def group(x):
        return dict(cases=len(x),median_case_paired_wall_ratio=median([r['paired_wall_ratio'] for r in x]),
            min_case_paired_wall_ratio=min(r['paired_wall_ratio'] for r in x),max_case_paired_wall_ratio=max(r['paired_wall_ratio'] for r in x),
            median_full_wall_ms=median([r['full']['wall_ms'] for r in x]),median_local_wall_ms=median([r['local']['wall_ms'] for r in x]),
            median_incremental_workspace_ratio=median([r['incremental_workspace_ratio'] for r in x]),
            median_full_peak_increment_bytes=median([r['full']['peak_increment_bytes'] for r in x]),median_local_peak_increment_bytes=median([r['local']['peak_increment_bytes'] for r in x]))
    groups={'all':group(summaries),'affected':group([r for r in summaries if r['affected_tiles']>0]),'zero_affected':group([r for r in summaries if r['affected_tiles']==0])}
    for scene in ['flowers','room']:groups[scene]=group([r for r in summaries if r['scene']==scene])
    views=[]
    for setup in r['setups']:
        selected=[x for x in summaries if x['scene']==setup['scene'] and x['camera']==setup['camera']];f=sum(x['full']['wall_ms'] for x in selected);l=sum(x['local']['wall_ms'] for x in selected);S=setup['setup_wall_ms']
        F=f/8;L=l/8;views.append(dict(**setup,full_8_prepared_ms=f,local_8_plus_observed_setup_ms=S+l,
            full_over_local_8_with_setup=f/(S+l),estimated_equal_mix_strict_break_even_queries=math.floor(S/(F-L))+1 if F>L else None,
            label='Observed one-time setup, including first-call effects; break-even assumes repeated equal8-event mix'))
    out=dict(all_verified=True,pair_count=256,case_summaries=summaries,groups=groups,views=views,
        setup_plus_32_total_ratio=sum(v['full_8_prepared_ms'] for v in views)/sum(v['local_8_plus_observed_setup_ms'] for v in views),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'plan.json',OUT/'results.json',Path(__file__)]},training=False,
        claim_scope='Prepared cached single-parent RGB/alpha/path query, not training/general inference; additional workspace above retained context')
    write_new(OUT/'independent_analysis.json',out);print(json.dumps(dict(groups=groups,views=views,setup_inclusive_total_ratio=out['setup_plus_32_total_ratio'])),flush=True)

if __name__=='__main__':main()
