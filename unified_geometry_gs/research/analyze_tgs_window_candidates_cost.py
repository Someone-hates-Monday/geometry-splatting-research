"""Frozen paired estimator, all calls retained; one device session only."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from tgs_window_candidates_common import ROOT,OUT,sha
from analyze_tgs_phase_cost import describe,bootstrap

COST=OUT/'cost'
KEYS=['forward_wall_ms','backward_wall_ms','pair_wall_ms','forward_event_ms','backward_event_ms']

def main():
    target=COST/'analysis.json';assert not target.exists()
    p=json.loads((COST/'plan.json').read_text());d=json.loads((COST/'results.json').read_text())
    assert d['plan_sha256']==sha(COST/'plan.json') and d['input_hashes_after']==p['model_before']
    variants=p['variants'];eligible=p['eligible'];rows=d['rows'];rng=np.random.default_rng(202610212)
    contrasts=[(v+'_oracle',v,'oracle') for v in eligible]+[(v+'_piecewise',v,'piecewise') for v in ['oracle']+eligible]
    pairs=[];cameras=[];gates={v:True for v in eligible}
    for ci in range(len(p['cameras'])):
        cr=[r for r in rows if r['camera']==ci];descriptive={};drift={};positions={}
        for v in variants:
            vr=sorted([r for r in cr if r['variant']==v],key=lambda r:r['block'])
            descriptive[v]={k:describe([r[k] for r in vr]) for k in KEYS}
            for k in ['peak_allocated_bytes','resident_bytes','incremental_peak_bytes','peak_reserved_bytes','tile_instances']:
                descriptive[v][k]=sorted(set(r[k] for r in vr))
            drift[v]=float(spearmanr([r['block'] for r in vr],[r['pair_wall_ms'] for r in vr]).statistic)
            positions[v]={str(j):describe([r['pair_wall_ms'] for r in vr if r['position']==j]) for j in range(len(variants))}
        blocks=sorted(set(r['block'] for r in cr));local=[]
        for bi in blocks:
            m={r['variant']:r for r in cr if r['block']==bi};assert set(m)==set(variants)
            row=dict(camera=ci,block=bi)
            for v in eligible:row[v+'_before_oracle']=m[v]['position']<m['oracle']['position']
            for label,num,den in contrasts:
                for k in KEYS:row[label+'_'+k]=m[num][k]/m[den][k]
            local.append(row);pairs.append(row)
        paired={label:{k:dict(**describe([r[label+'_'+k] for r in local]),paired_block_bootstrap_percentiles_95=bootstrap([r[label+'_'+k] for r in local],rng)) for k in KEYS} for label,_,_ in contrasts}
        order={v:{str(before):describe([r[v+'_oracle_pair_wall_ms'] for r in local if r[v+'_before_oracle']==before]) for before in [True,False]} for v in eligible}
        screen={}
        for v in eligible:
            q=paired[v+'_oracle']['pair_wall_ms'];screen[v]=bool(q['median']<=.90 and q['paired_block_bootstrap_percentiles_95'][1]<1);gates[v]&=screen[v]
        cameras.append(dict(camera=ci,image=p['cameras'][ci]['img_name'],split=p['cameras'][ci]['split'],descriptives=descriptive,paired_ratios=paired,order_sensitivity=order,position_wall_distributions=positions,pair_wall_spearman_over_block_order=drift,engineering_screen=screen))
    telemetry=json.loads((COST/'telemetry.json').read_text());dev=[]
    for t in [p['telemetry_start']]+[t for b in telemetry for t in [b['before'],b['after']]]+[d['telemetry_end']]:
        if t['returncode']!=0:continue
        vals=next(csv.reader([t['stdout']]));entry={}
        for k,v in zip(t['fields'].split(','),vals):
            try:entry[k]=float(v)
            except ValueError:pass
        dev.append(entry)
    ranges={k:dict(min=min(e[k] for e in dev if k in e),max=max(e[k] for e in dev if k in e)) for k in ['temperature.gpu','clocks.current.graphics','clocks.current.memory','memory.used','utilization.gpu'] if any(k in e for e in dev)}
    for name,rs in [('raw_calls.csv',rows),('paired_ratios.csv',pairs)]:
        with (COST/name).open('w',newline='') as fp:
            w=csv.DictWriter(fp,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    a=dict(cameras=cameras,measured_pairs=len(rows),paired_blocks=len(pairs),engineering_screen=gates,bootstrap_replicates=20000,bootstrap_seed=202610212,exclusions=[],all_samples_retained=True,telemetry_numeric_ranges=ranges,script_sha256=sha(Path(__file__)),plan_sha256=sha(COST/'plan.json'),results_sha256=sha(COST/'results.json'),analysis_plan_sha256=sha(OUT/'cost_analysis_plan.json'),estimator='Median of within-block candidate/oracle ratios, per camera.',uncertainty='Descriptive paired-block bootstrap conditional on one unlocked-clock session. Blocks may be temporally dependent; no scene/seed population inference or p values.',scope='Native API allocation and host synchronization included; not training or pure-kernel timing. Piecewise is a numerically different historical context.',training=False)
    target.write_text(json.dumps(a,indent=2)+'\n')
    print(json.dumps(dict(measured_pairs=len(rows),engineering_screen=gates,cameras=[dict(image=c['image'],primary={v:c['paired_ratios'][v+'_oracle']['pair_wall_ms'] for v in eligible},context={v:c['paired_ratios'][v+'_piecewise']['pair_wall_ms']['median'] for v in ['oracle']+eligible},median_pair_wall={v:c['descriptives'][v]['pair_wall_ms']['median'] for v in variants}) for c in cameras])),flush=True)

if __name__=='__main__':main()
