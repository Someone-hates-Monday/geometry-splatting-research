"""Recompute complete real-scene image/gradient/count summaries from saved arrays."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_adaptive_window_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=OUT/'real_independent_audit.json';assert not target.exists()
    plan=json.loads((OUT/'real_plan.json').read_text());r=json.loads((OUT/'real_results.json').read_text())
    assert plan['script_sha256']==sha(ROOT/'measure_tgs_adaptive_real.py') and r['plan_sha256']==sha(OUT/'real_plan.json')
    rows=[]
    for ci,record in enumerate(r['rows']):
        arrays={}
        for v in ['adaptive','oracle64']:
            with np.load(OUT/f'real_{ci}_{v}.npz') as z:arrays[v]={k:z[k].copy() for k in z.files}
            totals=arrays[v]['counters'].astype(np.uint64).sum(0).tolist();assert totals==record['metrics'][v]['counters']
            assert (arrays[v]['counters'][:,1:]<=arrays[v]['counters'][:,:1]).all()
            assert all(np.isfinite(a).all() for k,a in arrays[v].items() if k.startswith('gradient_'))
        a,b=arrays['adaptive'],arrays['oracle64'];d=np.abs(a['rgb'].astype(float)-b['rgb'].astype(float));ad=np.abs(a['aux'].astype(float)-b['aux'].astype(float))
        assert float(d.max())==record['rgb_max_abs'] and float(d.mean())==record['rgb_mae'] and int((d.max(0)>3e-5).sum())==record['pixels_rgb_delta_gt_3e5']
        assert np.array_equal(a['radii'],b['radii'])
        comp=[]
        for i in range(6):
            aa=a[f'gradient_{i}'].astype(float).ravel();bb=b[f'gradient_{i}'].astype(float).ravel();norm=float(np.linalg.norm(bb))
            rel=float(np.linalg.norm(aa-bb)/max(norm,1e-30));assert abs(rel-record['gradient_comparisons'][i]['relative'])<1e-12
            comp.append(rel)
        channels=[]
        for i,name in enumerate(['weighted_depth','alpha','normal_x','normal_y','normal_z','median_depth','distortion']):
            y,x=np.unravel_index(ad[i].argmax(),ad[i].shape)
            channels.append(dict(channel=i,name=name,max_abs=float(ad[i].max()),mean_abs=float(ad[i].mean()),pixels_delta_gt_3e5=int((ad[i]>3e-5).sum()),worst_pixel=[int(x),int(y)],adaptive=float(a['aux'][i,y,x]),oracle=float(b['aux'][i,y,x])))
        total,high,uncertain=record['metrics']['adaptive']['counters']
        rows.append(dict(camera=ci,image=record['image'],rgb_max_abs=float(d.max()),rgb_mae=float(d.mean()),aux_channels=channels,gradient_relative=comp,fallback_fraction=high/total,
            evaluated_pairs=total,high_precision_pairs=high,sign_uncertain_pairs=uncertain,window_evaluation_count_delta_vs_oracle=total-record['metrics']['oracle64']['counters'][0],
            complete_real_rgb_vjp_gate=record['gate'],compositing_branch_followup_required=any(c['pixels_delta_gt_3e5'] for c in channels)))
    result=dict(rows=rows,script_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'real_plan.json'),results_sha256=sha(OUT/'real_results.json'),all_declared_gates=all(r['complete_real_rgb_vjp_gate'] for r in rows),
        production_promotion_allowed=False,performance_measurement_allowed=False,
        reason='RGB and dense RGB VJP comparisons pass, but auxiliary channel differences and nonidentical traversal counts require compositing branch analysis before promotion. Oracle is a numerical reference, not GT.',training=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
