"""Audit all shared-cache timing samples and charge loss prep to both methods."""
from collections import Counter
import itertools,json,math
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,write_new
from benchmark_tgs_shared_loss_v26 import OUT,TIM,ARMS,V25

def median(x):return float(np.median(x))

def main():
    p=json.loads((TIM/'plan.json').read_text());r=json.loads((TIM/'results.json').read_text())
    assert r['native_forwards']==p['derived_control_forward_cap']==1004 and r['measured_endpoints']==128 and len(r['rows'])==4
    for path in [TIM/'plan.json',OUT/'plan.json',V25/'plan.json',V25/'runtime_dependency_addendum.json']:
        for n,h in json.loads(path.read_text())['inputs'].items():assert sha(ROOT/n)==h,n
    rows=[];nblocks=0;forward_count=0;allratios=[]
    for row,item in zip(r['rows'],p['views']):
        scene,view,K=row['scene'],row['view'],row['K'];assert (scene,view,K)==(item['scene'],item['view'],item['K']);key=f'{scene}_{view}'
        bs=row['blocks'];assert len(bs)==8 and [b['repeat'] for b in bs]==list(range(8)) and [b['order'] for b in bs]==p['orders'][key]
        assert row['setup']==json.loads((TIM/(key+'_setup.json')).read_text()) and row['model_inputs_unchanged']
        positions=Counter((a,i) for b in bs for i,a in enumerate(b['order']));neighbors=Counter((a,c) for b in bs for a,c in zip(b['order'],b['order'][1:]))
        assert all(positions[(a,i)]==2 for a in ARMS for i in range(4)) and all(neighbors[(a,b)]==2 for a,b in itertools.permutations(ARMS,2))
        forward_count+=1+p['warm_blocks']*(3*K+1)
        for b in bs:
            assert b==json.loads((TIM/(key+f"_block{b['repeat']}.json")).read_text());m=b['measurements'];assert set(m)==set(ARMS)
            for arm in ARMS:
                for field in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']:assert math.isfinite(m[arm][field]) and m[arm][field]>0
            ratios=dict(full_over_batch=m[ARMS[0]]['wall_ms']/m[ARMS[3]]['wall_ms'],cached_image_over_sequential=m[ARMS[1]]['wall_ms']/m[ARMS[2]]['wall_ms'],
                sequential_over_batch=m[ARMS[2]]['wall_ms']/m[ARMS[3]]['wall_ms'],cached_image_over_batch=m[ARMS[1]]['wall_ms']/m[ARMS[3]]['wall_ms'])
            assert ratios==b['ratios'];forward_count+=3*K+1;assert b['cumulative_native_forwards']==forward_count;nblocks+=1;allratios.append(ratios['full_over_batch'])
        assert row==json.loads((TIM/(key+'.json')).read_text())
        out=dict(scene=scene,view=view,K=K,ratios={name:median([b['ratios'][name] for b in bs]) for name in bs[0]['ratios']},
            methods={arm:{f:median([b['measurements'][arm][f] for b in bs]) for f in ['wall_ms','cuda_bracket_ms','peak_increment_bytes']} for arm in ARMS},
            slower_full_over_batch_blocks=sum(b['ratios']['full_over_batch']<1 for b in bs),setup=row['setup'])
        F=out['methods'][ARMS[0]]['wall_ms'];D=out['methods'][ARMS[3]]['wall_ms'];S=row['setup']['setup_wall_ms'];R=row['setup']['setup_render_ms'];L=row['setup']['setup_loss_ms']
        assert math.isclose(S,R+L,rel_tol=1e-12,abs_tol=1e-9) and R>0 and L>0 and row['setup']['cache_flags_zero']
        expected_cache_bytes=9*(1256*828 if scene=='flowers' else 1557*1038)*8
        assert row['setup']['shared_loss_cache_bytes']==expected_cache_bytes
        out.update(estimate_given_original_RGB_no_caches=(L+F)/(S+D),estimate_original_render_required_by_both=(S+F)/(S+D),
            estimate_given_original_RGB_and_loss_cache_no_geometry=F/(R+D),
            setup_estimate_scope='Same-process S_R and S_L; supplied RGB contract charges S_L to full too; arithmetic only, not measured cold batch')
        rows.append(out)
    assert forward_count==1004 and nblocks==32
    audit=dict(all_verified=True,rows=rows,blocks=nblocks,measured_endpoints=128,native_forwards=forward_count,
        all_block_full_over_batch_median=median(allratios),slower_full_over_batch_blocks=sum(x<1 for x in allratios),
        training=False,FP64_oracle_not_FP32_training=True,policy_quality_or_novelty_established=False,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in [TIM/'plan.json',TIM/'results.json',Path(__file__)]})
    write_new(TIM/'independent_analysis.json',audit);print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
