"""Audit balanced four-arm blocks and quantify matched-scope query/auxiliary effects."""
from collections import Counter
import itertools,json,math
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
from benchmark_tgs_aux_ablation_v21 import ARMS
OUT=ROOT/'experiments/tgs_aux_ablation_v21';TIM=OUT/'timing'

def med(x):return float(np.median(x))

def main():
    p=json.loads((TIM/'plan.json').read_text());r=json.loads((TIM/'results.json').read_text());v=json.loads((OUT/'validation/results.json').read_text())
    assert v['all_gates_passed'] and v['full_forwards']==204 and len(v['rows'])==48 and len(v['baselines'])==6
    assert r['full_forwards']==644 and len(r['rows'])==16
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    rows=[];matched_slower_blocks=0
    for row in r['rows']:
        key=f"{row['scene']}_p{row['parent_index']}_{row['appearance']}";blocks=row['blocks'];assert len(blocks)==8
        assert [b['order'] for b in blocks]==p['orders'][key] and [b['repeat'] for b in blocks]==list(range(8))
        positions=Counter((a,i) for b in blocks for i,a in enumerate(b['order']));neighbors=Counter((a,c) for b in blocks for a,c in zip(b['order'],b['order'][1:]))
        assert all(positions[(a,i)]==2 for a in ARMS for i in range(4));assert all(neighbors[(a,b)]==2 for a,b in itertools.permutations(ARMS,2))
        for block in blocks:
            assert set(block['measurements'])==set(ARMS)
            for a in ARMS:
                for field in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']:assert math.isfinite(block['measurements'][a][field]) and block['measurements'][a][field]>0
            m=block['measurements'];checks=dict(matched_full_over_local=m['B_full_rgb']['wall_ms']/m['D_local_rgbchild']['wall_ms'],
                native_full_over_control=m['A_full_aux']['wall_ms']/m['B_full_rgb']['wall_ms'],
                local_originalchild_over_controlchild=m['C_local_auxchild']['wall_ms']/m['D_local_rgbchild']['wall_ms'],
                original_full_over_original_local=m['A_full_aux']['wall_ms']/m['C_local_auxchild']['wall_ms'])
            assert checks==block['ratios'];matched_slower_blocks+=checks['matched_full_over_local']<1
        summary={k:row[k] for k in ['scene','parent_index','parent_id','appearance','coverage']}
        summary['ratios']={name:med([b['ratios'][name] for b in blocks]) for name in blocks[0]['ratios']}
        summary['methods']={a:{field:med([b['measurements'][a][field] for b in blocks]) for field in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']} for a in ARMS}
        summary['matched_local_slower_blocks']=sum(b['ratios']['matched_full_over_local']<1 for b in blocks)
        summary['matched_incremental_workspace_ratio']=summary['methods']['B_full_rgb']['peak_increment_bytes']/summary['methods']['D_local_rgbchild']['peak_increment_bytes'];rows.append(summary)
    def group(x):
        return dict(cases=len(x),median_matched_ratio=med([v['ratios']['matched_full_over_local'] for v in x]),
            min_matched_ratio=min(v['ratios']['matched_full_over_local'] for v in x),max_matched_ratio=max(v['ratios']['matched_full_over_local'] for v in x),
            matched_slower_case_medians=sum(v['ratios']['matched_full_over_local']<1 for v in x),
            median_full_aux_ablation_ratio=med([v['ratios']['native_full_over_control'] for v in x]),
            median_local_child_aux_ablation_ratio=med([v['ratios']['local_originalchild_over_controlchild'] for v in x]),
            median_old_full_over_local_ratio=med([v['ratios']['original_full_over_original_local'] for v in x]),
            median_matched_workspace_increment_ratio=med([v['matched_incremental_workspace_ratio'] for v in x]))
    groups={'all':group(rows)}
    for scene in ['flowers','room']:groups[scene]=group([v for v in rows if v['scene']==scene])
    for appearance in ['hold','moved']:groups[appearance]=group([v for v in rows if v['appearance']==appearance])
    views=[]
    for scene in ['flowers','room']:
        selected=[v for v in rows if v['scene']==scene];S=next(s for s in r['setups'] if s['scene']==scene and s['kind']=='B_control_cache');F=sum(v['methods']['B_full_rgb']['wall_ms'] for v in selected);L=sum(v['methods']['D_local_rgbchild']['wall_ms'] for v in selected)
        views.append(dict(**S,matched_full_8_sum_medians_ms=F,matched_local_8_plus_setup_ms=L+S['setup_wall_ms'],
            estimated_setup_inclusive_ratio=F/(L+S['setup_wall_ms']),estimated_equal_mix_break_even=math.floor(S['setup_wall_ms']/(F/8-L/8))+1 if F>L else None))
    out=dict(all_verified=True,rows=rows,groups=groups,views=views,blocks=128,measured_endpoints=512,matched_local_slower_blocks=matched_slower_blocks,
        estimated_total_setup_inclusive_matched_ratio=sum(v['matched_full_8_sum_medians_ms'] for v in views)/sum(v['matched_local_8_plus_setup_ms'] for v in views),
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [TIM/'plan.json',TIM/'results.json',OUT/'validation/results.json',Path(__file__)]},training=False,
        allocation_and_unused_preprocessing_ablated=False,cache_attribution_conditional_on_current_control=True)
    write_new(TIM/'independent_analysis.json',out);print(json.dumps(dict(groups=groups,views=views,slower_blocks=matched_slower_blocks,estimated_setup_inclusive_matched_ratio=out['estimated_total_setup_inclusive_matched_ratio'])),flush=True)

if __name__=='__main__':main()
