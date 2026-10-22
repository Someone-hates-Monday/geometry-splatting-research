"""Audit recovered complete blocks and conditional one-shot setup estimates."""
from collections import Counter
import itertools,json,math
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
from benchmark_tgs_batch_score_v22 import ARMS
OUT=ROOT/'experiments/tgs_batch_score_v22';TIM=OUT/'timing'

def med(x):return float(np.median(x))

def main():
    p=json.loads((TIM/'plan.json').read_text());r=json.loads((TIM/'results_recovered.json').read_text());recovery=json.loads((TIM/'recovery_plan.json').read_text());setup=json.loads((TIM/'setup_addendum.json').read_text())
    assert r['full_forwards']==982 and r['original_run_exit_code']==1 and recovery['derived_actual_forwards']==982
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    for n,h in recovery['inputs'].items():assert sha(ROOT/n)==h,n
    source=json.loads((OUT/'results.json').read_text());assert source['all_gates_passed'] and json.loads((OUT/'independent_audit.json').read_text())['all_passed']
    rows=[];blocks=0;measurements=0
    for row in r['rows']:
        scene,K=row['scene'],row['K'];key=f'{scene}_K{K}';bs=row['blocks'];assert len(bs)==8 and [b['repeat'] for b in bs]==list(range(8)) and [b['order'] for b in bs]==p['orders'][key]
        positions=Counter((a,i) for b in bs for i,a in enumerate(b['order']));neighbors=Counter((a,c) for b in bs for a,c in zip(b['order'],b['order'][1:]))
        assert all(positions[(a,i)]==2 for a in ARMS for i in range(4)) and all(neighbors[(a,b)]==2 for a,b in itertools.permutations(ARMS,2))
        for b in bs:
            m=b['measurements'];assert set(m)==set(ARMS)
            for arm in ARMS:
                for field in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']:assert math.isfinite(m[arm][field]) and m[arm][field]>0
            expected=dict(full_over_batch=m[ARMS[0]]['wall_ms']/m[ARMS[3]]['wall_ms'],cached_image_over_sequential_scalar=m[ARMS[1]]['wall_ms']/m[ARMS[2]]['wall_ms'],
                sequential_scalar_over_batch=m[ARMS[2]]['wall_ms']/m[ARMS[3]]['wall_ms'],cached_image_over_batch=m[ARMS[1]]['wall_ms']/m[ARMS[3]]['wall_ms'])
            assert expected==b['ratios'];blocks+=1;measurements+=4
        out=dict(scene=scene,K=K,proposals=row['proposals'],ratios={n:med([b['ratios'][n] for b in bs]) for n in bs[0]['ratios']},
            methods={a:{f:med([b['measurements'][a][f] for b in bs]) for f in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']} for a in ARMS},
            slower_full_over_batch_blocks=sum(b['ratios']['full_over_batch']<1 for b in bs))
        if K==8:
            S=next(x['setup_wall_ms'] for x in setup['rows'] if x['scene']==scene);F=out['methods']['A_full_score']['wall_ms'];D=out['methods']['D_batch_scalar']['wall_ms']
            out['separate_setup_ms']=S;out['estimate_given_original_PNG_no_geometry_cache']=F/(S+D);out['estimate_original_render_required_by_both']=(S+F)/(S+D)
            out['setup_estimate_caveat']='Separate two-forward addendum; original setup/telemetry not persisted; arithmetic only'
        rows.append(out)
    assert blocks==64 and measurements==256 and len(rows)==8
    out=dict(all_verified=True,rows=rows,blocks=blocks,measured_endpoints=measurements,original_run_exit_code=1,recovery_complete_raw_cells=True,
        original_setup_and_final_telemetry_missing=True,setup_addendum_separate_process=True,actual_timing_forwards=982,training=False,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [TIM/'results_recovered.json',TIM/'plan.json',TIM/'recovery_plan.json',TIM/'setup_addendum.json',Path(__file__)]},
        no_GPU_timing_rerun=True,GT_policy_or_quality_improvement_established=False)
    write_new(TIM/'independent_analysis.json',out);print(json.dumps(dict(rows=[v for v in rows if v['K']==8],blocks=blocks,original_run_exit_code=1)),flush=True)

if __name__=='__main__':main()
