"""Audit schedule, raw metrics, paired estimator and bootstrap reproducibility."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_phase_cost_v5';COST=OUT/'cost'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def median(x):
    a=sorted(x);n=len(a);return (a[(n-1)//2]+a[n//2])/2
def main():
    target=COST/'independent_audit.json';assert not target.exists();p=json.loads((COST/'plan.json').read_text());d=json.loads((COST/'results.json').read_text());a=json.loads((COST/'analysis.json').read_text());precision=json.loads((OUT/'precision_audit.json').read_text())
    assert d['plan_sha256']==sha(COST/'plan.json')==a['plan_sha256'] and a['results_sha256']==sha(COST/'results.json')
    assert p['script_sha256']==sha(ROOT/'benchmark_tgs_phase_cost.py') and a['script_sha256']==sha(ROOT/'analyze_tgs_phase_cost.py') and p['precision_audit_sha256']==sha(OUT/'precision_audit.json')
    assert precision['all_gates'] and precision['performance_measurement_allowed'] and d['input_hashes_after']==p['input_tensor_hashes']
    assert len(p['balanced_blocks'])==72 and len(d['rows'])==216 and len(d['guards'])==12
    assert all(g['all_gradients_finite'] and g['forward_matches_validation_or_history'] for g in d['guards'])
    for v,r in p['modules'].items():
        assert not r['instrumented'] and sha(Path(r['module_file']))==r['module_sha256']
        if v!='piecewise':assert not json.loads((OUT/v/'build_receipt.json').read_text())['instrumented']
    for i,block in enumerate(p['balanced_blocks']):
        rows=[r for r in d['rows'] if r['block']==i];assert len(rows)==3
        rows.sort(key=lambda r:r['position']);assert [r['variant'] for r in rows]==block['variants'] and all(r['camera']==block['camera'] and r['repeat']==block['repeat'] for r in rows)
    keys=['forward_wall_ms','backward_wall_ms','pair_wall_ms','forward_event_ms','backward_event_ms'];rng=np.random.default_rng(202610207);summaries=[]
    for ci in range(4):
        camera=a['cameras'][ci];assert camera['image']==p['cameras'][ci]['img_name'];cr=[r for r in d['rows'] if r['camera']==ci]
        for v in ['piecewise','guard','oracle']:
            selected=[r for r in cr if r['variant']==v];assert len(selected)==18
            assert [sum(r['position']==j for r in selected) for j in range(3)]==[6,6,6]
            assert len(set(r['tile_instances'] for r in selected))==1
            for row in selected:
                assert all(np.isfinite(row[k]) and row[k]>0 for k in keys)
                assert row['pair_wall_ms']+1e-7>=row['forward_wall_ms']+row['backward_wall_ms']
                assert row['peak_allocated_bytes']-row['resident_bytes']==row['incremental_peak_bytes']
            for key in keys:assert abs(median([r[key] for r in selected])-camera['descriptives'][v][key]['median'])<1e-12
        blocks=sorted(set(r['block'] for r in cr));ratios={c:{key:[] for key in keys} for c in ['guard_oracle','guard_piecewise','oracle_piecewise']}
        for block in blocks:
            m={r['variant']:r for r in cr if r['block']==block}
            for contrast,numerator,denominator in [('guard_oracle','guard','oracle'),('guard_piecewise','guard','piecewise'),('oracle_piecewise','oracle','piecewise')]:
                for key in keys:ratios[contrast][key].append(m[numerator][key]/m[denominator][key])
        for contrast in ratios:
            for key,values in ratios[contrast].items():
                rr=camera['paired_ratios'][contrast][key];assert abs(median(values)-rr['median'])<1e-12 and rr['n']==18 and rr['missing']==0
                x=np.asarray(values);indices=rng.integers(0,18,size=(20000,18));ordered=np.partition(x[indices],[8,9],axis=1);draw=(ordered[:,8]+ordered[:,9])/2;interval=np.quantile(draw,[.025,.975]).tolist()
                assert np.allclose(interval,rr['paired_block_bootstrap_percentiles_95'],rtol=0,atol=1e-12)
        primary=camera['paired_ratios']['guard_oracle']['pair_wall_ms'];summaries.append(dict(camera=ci,image=camera['image'],paired_guard_oracle_wall=primary['median'],descriptive_interval=primary['paired_block_bootstrap_percentiles_95'],median_guard_old_piecewise=camera['paired_ratios']['guard_piecewise']['pair_wall_ms']['median'],no_observed_median_acceleration=primary['median']>=1))
    audit=dict(primary=summaries,all_raw_rows_finite_positive=True,all_samples_retained=True,paired_design_and_order_balanced=True,input_tensors_unchanged=True,bootstrap_independently_recomputed=True,measured_pairs=216,
        all_four_median_ratios_above_one=all(s['no_observed_median_acceleration'] for s in summaries),efficiency_promotion_supported=False,
        reason='No observed median acceleration over the correct oracle on any of the four views. Unlocked clocks and temporally dependent blocks limit inference; correctness evidence is retained separately.',
        plan_sha256=sha(COST/'plan.json'),results_sha256=sha(COST/'results.json'),analysis_sha256=sha(COST/'analysis.json'),script_sha256=sha(Path(__file__)),training=False)
    target.write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
