"""Independent paired cost and two-scene schedule audit, with precision prerequisite."""
import itertools,json,math
from pathlib import Path
import numpy as np
from tgs_cross_scene_common_v10 import ROOT,OUT,sha

COST=OUT/'cost';KEYS=['forward_wall_ms','backward_wall_ms','pair_wall_ms','forward_event_ms','backward_event_ms']
def median(a):
    a=sorted(a);return (a[(len(a)-1)//2]+a[len(a)//2])/2

def main():
    target=COST/'independent_audit.json';assert not target.exists();p=json.loads((COST/'plan.json').read_text());d=json.loads((COST/'results.json').read_text());a=json.loads((COST/'analysis.json').read_text());f=json.loads((OUT/'cost_analysis_plan.json').read_text());precision=json.loads((OUT/'precision_audit.json').read_text());assert precision['all_gates'] and precision['performance_measurement_allowed']
    for name,h in f['scripts'].items():assert sha(ROOT/name)==h
    assert not f['timing_started'] and a['analysis_plan_sha256']==sha(OUT/'cost_analysis_plan.json')
    assert p['script_sha256']==sha(ROOT/'benchmark_tgs_cross_scene_v10.py') and p['protocol_sha256']==sha(ROOT/'TGS_CROSS_SCENE_COST_PROTOCOL_V10.md') and p['precision_audit_sha256']==sha(OUT/'precision_audit.json')
    assert a['script_sha256']==sha(ROOT/'analyze_tgs_cross_scene_cost_v10.py') and a['results_sha256']==sha(COST/'results.json') and a['plan_sha256']==d['plan_sha256']==sha(COST/'plan.json');assert d['input_hashes_after']==p['model_before'] and d['inputs_unchanged'] and not d['training']
    for v,r in p['modules'].items():assert sha(Path(r['path']))==r['sha256']
    variants=['oracle','coarse','piecewise'];assert p['variants']==variants and p['eligible']==['coarse'] and p['scenes']==['flowers','room'];rng=np.random.default_rng(202610213);orders=list(itertools.permutations(variants));schedule=[]
    for si,scene in enumerate(p['scenes']):
        for repeat in range(3):
            batch=[dict(scene=scene,camera=2*si+ci,local_camera=ci,repeat=repeat,order_index=oi,variants=list(o)) for ci in range(2) for oi,o in enumerate(orders)];rng.shuffle(batch);schedule.extend(batch)
    assert p['blocks']==schedule and len(schedule)==72 and len(d['rows'])==216 and p['warmups']==3
    guards=json.loads((COST/'preflight_guards.json').read_text());assert len(guards)==8 and all(all(g['equal'].values()) for g in guards);assert len(json.loads((COST/'telemetry.json').read_text()))==72
    for bi,b in enumerate(schedule):
        rs=sorted([r for r in d['rows'] if r['block']==bi],key=lambda r:r['position']);assert len(rs)==3 and [r['variant'] for r in rs]==b['variants'] and all(r['camera']==b['camera'] and r['scene']==b['scene'] for r in rs)
    rng=np.random.default_rng(202610214);summaries=[];all_pass=True
    for ci in range(4):
        cr=[r for r in d['rows'] if r['camera']==ci];ca=a['cameras'][ci];assert ca['image']==p['cameras'][ci]['img_name'] and ca['scene']==p['cameras'][ci]['scene']
        for v in variants:
            vr=[r for r in cr if r['variant']==v];assert len(vr)==18 and [sum(r['position']==j for r in vr) for j in range(3)]==[6,6,6];assert len(set(r['tile_instances'] for r in vr))==1
            for r in vr:
                assert all(math.isfinite(r[k]) and r[k]>0 for k in KEYS);assert r['pair_wall_ms']+1e-7>=r['forward_wall_ms']+r['backward_wall_ms'];assert r['peak_allocated_bytes']-r['resident_bytes']==r['incremental_peak_bytes']
            for k in KEYS:assert abs(median([r[k] for r in vr])-ca['descriptives'][v][k]['median'])<1e-12
        ids=sorted(set(r['block'] for r in cr));paired={bi:{r['variant']:r for r in cr if r['block']==bi} for bi in ids}
        for label,num,den in [('coarse_oracle','coarse','oracle'),('oracle_piecewise','oracle','piecewise'),('coarse_piecewise','coarse','piecewise')]:
            for k in KEYS:
                x=np.array([paired[bi][num][k]/paired[bi][den][k] for bi in ids]);rr=ca['paired_ratios'][label][k];assert rr['n']==18 and rr['missing']==0 and abs(median(x)-rr['median'])<1e-12;ix=rng.integers(0,18,size=(20000,18));part=np.partition(x[ix],[8,9],axis=1);draw=(part[:,8]+part[:,9])/2;bounds=np.quantile(draw,[.025,.975]);assert np.allclose(bounds,rr['paired_block_bootstrap_percentiles_95'],atol=1e-12,rtol=0)
        q=ca['paired_ratios']['coarse_oracle']['pair_wall_ms'];passed=q['median']<=.90 and q['paired_block_bootstrap_percentiles_95'][1]<1;assert passed==ca['engineering_screen']['coarse'];all_pass &= passed;summaries.append(dict(scene=ca['scene'],image=ca['image'],primary=q,engineering_screen=passed))
    assert a['engineering_screen']=={'coarse':all_pass} and a['all_samples_retained'] and a['exclusions']==[];out=dict(all_verified=True,primary=summaries,engineering_screen={'coarse':all_pass},measured_pairs=216,balanced_blocks=72,input_tensors_unchanged=True,bootstrap_independently_recomputed=True,script_sha256=sha(Path(__file__)),analysis_sha256=sha(COST/'analysis.json'),results_sha256=sha(COST/'results.json'),plan_sha256=sha(COST/'plan.json'),training=False,scientific_goal_complete=False);target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='primary'}),flush=True)

if __name__=='__main__':main()
