"""Audit fixed-code stress measurements without dropping slow pairs or fitting routing."""
import json,math
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
OUT=ROOT/'experiments/tgs_local_stress_v20/timing'

def med(x):return float(np.median(x))

def main():
    p=json.loads((OUT/'plan.json').read_text());r=json.loads((OUT/'results.json').read_text());assert r['full_forwards']==322 and len(r['rows'])==16
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    rows=[]
    for row in r['rows']:
        key=f"{row['scene']}_p{row['parent_index']}_{row['appearance']}";pairs=row['pairs'];assert len(pairs)==8 and [x['repeat'] for x in pairs]==list(range(8))
        assert [x['order'] for x in pairs]==['full_then_local' if x==0 else 'local_then_full' for x in p['orders'][key]]
        assert sum(x['order']=='full_then_local' for x in pairs)==4
        for pair in pairs:
            for method in ['full','local']:
                for field in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']:assert math.isfinite(pair[method][field]) and pair[method][field]>0
            assert pair['full_over_local_wall']==pair['full']['wall_ms']/pair['local']['wall_ms']
        out={k:row[k] for k in ['scene','parent_index','parent_id','appearance','coverage','candidate_work_fraction']}
        out['paired_wall_ratio']=med([x['full_over_local_wall'] for x in pairs])
        for method in ['full','local']:out[method]={field:med([x[method][field] for x in pairs]) for field in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']}
        out['workspace_increment_ratio']=out['full']['peak_increment_bytes']/out['local']['peak_increment_bytes'];rows.append(out)
    def group(x):return dict(cases=len(x),median_case_paired_wall_ratio=med([v['paired_wall_ratio'] for v in x]),min_case_ratio=min(v['paired_wall_ratio'] for v in x),
        max_case_ratio=max(v['paired_wall_ratio'] for v in x),local_slower_cases=sum(v['paired_wall_ratio']<1 for v in x),
        median_workspace_increment_ratio=med([v['workspace_increment_ratio'] for v in x]),min_coverage=min(v['coverage'] for v in x),max_coverage=max(v['coverage'] for v in x))
    groups={'all':group(rows)}
    for scene in ['flowers','room']:groups[scene]=group([v for v in rows if v['scene']==scene])
    for appearance in ['hold','moved']:groups[appearance]=group([v for v in rows if v['appearance']==appearance])
    views=[]
    for setup in r['setups']:
        selected=[v for v in rows if v['scene']==setup['scene']];f=sum(v['full']['wall_ms'] for v in selected);l=sum(v['local']['wall_ms'] for v in selected);S=setup['setup_wall_ms']
        views.append(dict(**setup,full_8_sum_medians_ms=f,local_8_sum_medians_plus_setup_ms=l+S,estimated_setup_inclusive_ratio=f/(l+S),
            estimated_equal_mix_break_even=math.floor(S/(f/8-l/8))+1 if f>l else None))
    out=dict(all_verified=True,pair_count=128,rows=rows,groups=groups,views=views,
        estimated_total_setup_inclusive_ratio=sum(v['full_8_sum_medians_ms'] for v in views)/sum(v['local_8_sum_medians_plus_setup_ms'] for v in views),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'plan.json',OUT/'results.json',Path(__file__)]},
        training=False,routing_fitted=False,matched_output_full_scene_baseline_not_yet_tested=True)
    write_new(OUT/'independent_analysis.json',out);print(json.dumps(dict(groups=groups,views=views,estimated_total_setup_inclusive_ratio=out['estimated_total_setup_inclusive_ratio'])),flush=True)

if __name__=='__main__':main()
