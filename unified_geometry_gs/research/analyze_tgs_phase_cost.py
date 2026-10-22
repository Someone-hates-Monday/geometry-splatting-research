"""Paired within-session summaries; preserve every sample and order diagnostic."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_phase_cost_v5/cost'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(x):
    a=np.asarray(x,float);q1,q3=np.quantile(a,[.25,.75]);mask=(a<q1-1.5*(q3-q1))|(a>q3+1.5*(q3-q1))
    return dict(n=len(a),mean=float(a.mean()),sd=float(a.std(ddof=1)),median=float(np.median(a)),minimum=float(a.min()),maximum=float(a.max()),q25=float(q1),q75=float(q3),IQR_outlier_indices=np.flatnonzero(mask).tolist(),missing=int((~np.isfinite(a)).sum()))
def bootstrap(x,rng):
    a=np.asarray(x,float);samples=np.median(a[rng.integers(0,len(a),size=(20000,len(a)))],axis=1)
    return np.quantile(samples,[.025,.975]).tolist()
def main():
    target=OUT/'analysis.json';assert not target.exists();plan=json.loads((OUT/'plan.json').read_text());data=json.loads((OUT/'results.json').read_text());assert data['plan_sha256']==sha(OUT/'plan.json') and data['input_hashes_after']==plan['input_tensor_hashes']
    rows=data['rows'];assert len(rows)==216;keys=['forward_wall_ms','backward_wall_ms','pair_wall_ms','forward_event_ms','backward_event_ms'];variants=['piecewise','guard','oracle'];rng=np.random.default_rng(202610207);cameras=[];pairs=[]
    for ci in range(4):
        cr=[r for r in rows if r['camera']==ci];descriptives={};drift={}
        for v in variants:
            selected=sorted([r for r in cr if r['variant']==v],key=lambda r:r['block']);assert len(selected)==18
            descriptives[v]={key:describe([r[key] for r in selected]) for key in keys}
            descriptives[v]['peak_allocated_bytes']=sorted(set(r['peak_allocated_bytes'] for r in selected))
            descriptives[v]['incremental_peak_bytes']=sorted(set(r['incremental_peak_bytes'] for r in selected))
            drift[v]=float(spearmanr([r['block'] for r in selected],[r['pair_wall_ms'] for r in selected]).statistic)
        blocks=sorted(set(r['block'] for r in cr));paired={}
        for block in blocks:
            br=[r for r in cr if r['block']==block];assert len(br)==3 and sorted(r['variant'] for r in br)==sorted(variants);m={r['variant']:r for r in br}
            row=dict(camera=ci,block=block,guard_before_oracle=m['guard']['position']<m['oracle']['position'])
            for key in keys:
                row['guard_oracle_'+key]=m['guard'][key]/m['oracle'][key]
                row['guard_piecewise_'+key]=m['guard'][key]/m['piecewise'][key]
                row['oracle_piecewise_'+key]=m['oracle'][key]/m['piecewise'][key]
            pairs.append(row)
        cp=[r for r in pairs if r['camera']==ci]
        for contrast in ['guard_oracle','guard_piecewise','oracle_piecewise']:
            paired[contrast]={}
            for key in keys:
                x=[r[contrast+'_'+key] for r in cp]
                paired[contrast][key]=dict(**describe(x),paired_block_bootstrap_percentiles_95=bootstrap(x,rng))
        order={str(before):describe([r['guard_oracle_pair_wall_ms'] for r in cp if r['guard_before_oracle']==before]) for before in [True,False]}
        cameras.append(dict(camera=ci,image=plan['cameras'][ci]['img_name'],split=plan['cameras'][ci]['split'],descriptives=descriptives,paired_ratios=paired,guard_oracle_order_sensitivity=order,pair_wall_spearman_over_block_order=drift))
    def device_numeric(t):
        if t.get('returncode')!=0:return {}
        fields=t['fields'].split(',');values=next(csv.reader([t['stdout']]))
        if len(fields)!=len(values):return {}
        out={}
        for k,v in zip(fields,values):
            try:out[k]=float(v)
            except ValueError:pass
        return out
    device=[device_numeric(plan['telemetry_start'])]+[device_numeric(r['telemetry']) for r in data['device_samples']]+[device_numeric(data['telemetry_end'])]
    ranges={k:dict(min=min(v[k] for v in device if k in v),max=max(v[k] for v in device if k in v)) for k in ['temperature.gpu','clocks.current.graphics','clocks.current.memory','memory.used'] if any(k in v for v in device)}
    with (OUT/'raw_calls.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'paired_ratios.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    report=dict(cameras=cameras,measured_pairs=len(rows),paired_blocks=len(pairs),bootstrap_replicates=20000,bootstrap_seed=202610207,script_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'plan.json'),results_sha256=sha(OUT/'results.json'),numpy=np.__version__,scipy=scipy.__version__,telemetry_numeric_ranges=ranges,
        exclusions=[],all_samples_retained=True,estimator='median of within-block paired ratios, separately per camera; raw descriptive medians also shown',
        uncertainty='Paired-block percentile bootstrap describing this single device session; blocks may be temporally dependent. No p values or independent-scene/seed inference.',
        scope='Native API wall/event spans including allocation/.item sync, not pure kernel or training latency. Piecewise has a different numerical forward. Reserved counter allocations remain in lean versions.',training=False)
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(measured_pairs=len(rows),telemetry=ranges,cameras=[dict(image=c['image'],median_pair_wall={v:c['descriptives'][v]['pair_wall_ms']['median'] for v in variants},paired_guard_oracle=c['paired_ratios']['guard_oracle']['pair_wall_ms'],paired_guard_piecewise=c['paired_ratios']['guard_piecewise']['pair_wall_ms']) for c in cameras])),flush=True)
if __name__=='__main__':main()
