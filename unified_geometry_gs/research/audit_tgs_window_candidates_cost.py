"""Independent schedule reconstruction and paired-bootstrap calculation, no shared estimator."""
import itertools,json,math
from pathlib import Path
import numpy as np
from tgs_window_candidates_common import ROOT,OUT,sha

COST=OUT/'cost'
KEYS=['forward_wall_ms','backward_wall_ms','pair_wall_ms','forward_event_ms','backward_event_ms']
def med(x):
    a=sorted(x);return (a[(len(a)-1)//2]+a[len(a)//2])/2

def main():
    target=COST/'independent_audit.json';assert not target.exists()
    p=json.loads((COST/'plan.json').read_text());d=json.loads((COST/'results.json').read_text());a=json.loads((COST/'analysis.json').read_text());precision=json.loads((OUT/'precision_audit.json').read_text());freeze=json.loads((OUT/'cost_analysis_plan.json').read_text())
    for name,h in freeze['scripts'].items():assert sha(ROOT/name)==h
    assert freeze['timing_started']==False and a['analysis_plan_sha256']==sha(OUT/'cost_analysis_plan.json')
    assert p['script_sha256']==sha(ROOT/'benchmark_tgs_window_candidates.py') and p['protocol_sha256']==sha(ROOT/'TGS_WINDOW_COST_PROTOCOL_V9.md')
    assert p['precision_audit_sha256']==sha(OUT/'precision_audit.json') and a['script_sha256']==sha(ROOT/'analyze_tgs_window_candidates_cost.py')
    assert d['plan_sha256']==a['plan_sha256']==sha(COST/'plan.json') and a['results_sha256']==sha(COST/'results.json')
    eligible=[v for v in ['coarse','standard'] if precision['performance_measurement_allowed'][v]];variants=['oracle']+eligible+['piecewise'];assert variants==p['variants'] and eligible==p['eligible'] and eligible
    for v,r in p['modules'].items():assert sha(Path(r['path']))==r['sha256']
    assert d['input_hashes_after']==p['model_before'] and d['inputs_unchanged'] and not d['training']
    rng=np.random.default_rng(202610211);orders=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]] if len(variants)==4 else list(itertools.permutations(range(3)));schedule=[]
    for repeat in range(3):
        batch=[dict(camera=ci,repeat=repeat,order_index=oi,variants=[variants[i] for i in o]) for ci in range(7) for oi,o in enumerate(orders)];rng.shuffle(batch);schedule.extend(batch)
    assert p['blocks']==schedule and p['warmups']==3
    rows=d['rows'];assert len(rows)==len(schedule)*len(variants)
    assert len(json.loads((COST/'telemetry.json').read_text()))==len(schedule)
    guards=json.loads((COST/'preflight_guards.json').read_text());assert len(guards)==7*(1+len(eligible)) and all(all(g['equal'].values()) for g in guards)
    for bi,block in enumerate(schedule):
        rs=[r for r in rows if r['block']==bi];assert len(rs)==len(variants);rs.sort(key=lambda r:r['position']);assert [r['variant'] for r in rs]==block['variants'] and all(r['camera']==block['camera'] for r in rs)
    rng=np.random.default_rng(202610212);summary=[];overall={v:True for v in eligible}
    contrasts=[(v+'_oracle',v,'oracle') for v in eligible]+[(v+'_piecewise',v,'piecewise') for v in ['oracle']+eligible]
    for ci in range(7):
        cr=[r for r in rows if r['camera']==ci];ca=a['cameras'][ci];assert ca['image']==p['cameras'][ci]['img_name'];n=3*len(orders)
        for v in variants:
            vr=[r for r in cr if r['variant']==v];assert len(vr)==n
            assert [sum(r['position']==j for r in vr) for j in range(len(variants))]==[n//len(variants)]*len(variants)
            assert len(set(r['tile_instances'] for r in vr))==1
            for r in vr:
                assert all(math.isfinite(r[k]) and r[k]>0 for k in KEYS)
                assert r['pair_wall_ms']+1e-7>=r['forward_wall_ms']+r['backward_wall_ms']
                assert r['incremental_peak_bytes']==r['peak_allocated_bytes']-r['resident_bytes']
            for k in KEYS:assert abs(med([r[k] for r in vr])-ca['descriptives'][v][k]['median'])<1e-12
        bis=sorted(set(r['block'] for r in cr));byblock={bi:{r['variant']:r for r in cr if r['block']==bi} for bi in bis}
        for label,num,den in contrasts:
            for k in KEYS:
                x=np.array([byblock[bi][num][k]/byblock[bi][den][k] for bi in bis]);report=ca['paired_ratios'][label][k];assert report['n']==n and report['missing']==0 and abs(med(x)-report['median'])<1e-12
                indices=rng.integers(0,n,size=(20000,n));part=np.partition(x[indices],[(n-1)//2,n//2],axis=1);sample=(part[:,(n-1)//2]+part[:,n//2])/2;bounds=np.quantile(sample,[.025,.975]);assert np.allclose(bounds,report['paired_block_bootstrap_percentiles_95'],rtol=0,atol=1e-12)
        screens={}
        for v in eligible:
            q=ca['paired_ratios'][v+'_oracle']['pair_wall_ms'];screens[v]=q['median']<=.90 and q['paired_block_bootstrap_percentiles_95'][1]<1;assert screens[v]==ca['engineering_screen'][v];overall[v]&=screens[v]
        summary.append(dict(camera=ci,image=ca['image'],screen=screens,primary={v:ca['paired_ratios'][v+'_oracle']['pair_wall_ms'] for v in eligible}))
    assert overall==a['engineering_screen'] and a['all_samples_retained'] and a['exclusions']==[]
    result=dict(all_verified=True,engineering_screen=overall,primary=summary,measured_pairs=len(rows),balanced_blocks=len(schedule),bootstrap_independently_recomputed=True,input_tensors_unchanged=True,failures_excluded_only_by_precision=[v for v in ['coarse','standard'] if v not in eligible],scientific_goal_complete=False,training=False,script_sha256=sha(Path(__file__)),analysis_sha256=sha(COST/'analysis.json'),results_sha256=sha(COST/'results.json'),plan_sha256=sha(COST/'plan.json'))
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='primary'}),flush=True)

if __name__=='__main__':main()
