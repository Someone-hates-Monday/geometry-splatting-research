"""CPU audit of all isolated windows and their conditional comparison to frozen full-scene G."""
import json,math
from pathlib import Path
import numpy as np
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,exact
OLD=ROOT/'experiments/tgs_split_color_v14';OUT=ROOT/'experiments/tgs_window_scene_v15'

def reconstruct(path,record):
    assert sha(path)==record['file_sha256'];raw=np.load(path);h,w=raw['shape'];x0,y0,x1,y1=raw['bbox']
    assert list(raw['shape'])==record['shape'] and list(raw['bbox'])==record['bbox']
    assert 0<=x0<=x1<=w and 0<=y0<=y1<=h
    arrays={}
    for name in ['alpha','rgb']:
        a=np.zeros((h,w),dtype=raw[name].dtype);a[y0:y1,x0:x1]=raw[name]
        assert ahash(a)==record[name+'_sha256'] and str(a.dtype)==record['dtype']
        assert np.isfinite(a).all();arrays[name]=a
    assert raw['radii'].tolist()==record['radii']
    error=float(np.max(np.abs(arrays['alpha'].astype(float)-arrays['rgb'].astype(float))))
    assert error<=2e-6 and error==record['white_alpha_maxabs']
    return arrays

def rms(a,denominator):return math.sqrt(float(np.einsum('...,...->',a,a))/denominator)

def main():
    target=OUT/'independent_audit.json';assert not target.exists()
    plan=json.loads((OUT/'plan.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert result['plan_sha256']==sha(OUT/'plan.json') and result['forwards']==64
    assert result['all_radii_and_white_controls'] and len(result['cases'])==32
    assert result['script_sha256']==sha(ROOT/'render_tgs_window_scene_v15.py')
    for n,h in plan['scripts'].items():assert sha(ROOT/n)==h
    for n,h in plan['old_inputs'].items():assert sha(OLD/n)==h
    prior=json.loads((OLD/'results.json').read_text());rows=[];color_cache={}
    for row in result['cases']:
        case=next(r for r in prior['cases'] if all(r[k]==row[k] for k in ['scene','event','camera','parent_id']))
        base=OUT/row['scene']/f"event_{row['event']}"/f"camera_{row['camera']}"
        assert json.loads((base/'receipt.json').read_text())==row
        parent=reconstruct(base/'parent.npz',row['versions']['parent'])
        child=reconstruct(base/'children.npz',row['versions']['children'])
        assert row['versions']['parent']['radii']==[case['parent_radii']]
        assert row['versions']['children']['radii']==case['child_radii']
        P=parent['alpha'].astype(float);Q=child['alpha'].astype(float);delta=Q-P;h,w=P.shape
        oldbase=OLD/row['scene'];change=np.load(oldbase/f"event_{row['event']}/camera_{row['camera']}/changes.npz")
        x0,y0,x1,y1=change['bbox'];G=change['hold_rgb'].astype(float)-change['before_rgb'].astype(float)
        g_rms=math.sqrt(float(np.sum(G*G))/(3*h*w));assert abs(g_rms-case['metrics']['noncolor']['RMS'])<=1e-12
        key=(row['scene'],row['camera'])
        if key not in color_cache:
            colors=np.load(oldbase/f"baseline_{row['camera']}/colors.npy",mmap_mode='r')
            assert np.isfinite(colors).all() and np.min(colors)>=0
            color_cache[key]=np.max(colors,axis=0).astype(float)
        Cmax=color_cache[key]
        bound=Cmax[:,None,None]*np.abs(delta[y0:y1,x0:x1])[None]
        exceed=np.abs(G)-bound
        violations=np.any(exceed>3e-5,axis=0)
        delta_rms=math.sqrt(float(np.sum(delta*delta))/(h*w))
        p_sum=float(np.sum(P));q_sum=float(np.sum(Q))
        rows.append(dict(scene=row['scene'],event=row['event'],camera=row['camera'],parent_id=row['parent_id'],
            isolated_alpha_RMS=delta_rms,isolated_alpha_MAE=float(np.mean(np.abs(delta))),
            isolated_alpha_peak=float(np.max(np.abs(delta))),isolated_alpha_sum_parent=p_sum,
            isolated_alpha_sum_children=q_sum,relative_discrete_sum_change=q_sum/p_sum-1 if p_sum>1e-14 else None,
            delta_positive_pixels=int(np.count_nonzero(delta>0)),delta_negative_pixels=int(np.count_nonzero(delta<0)),
            delta_zero_pixels=int(np.count_nonzero(delta==0)),pixels_above_3e_minus5=int(np.count_nonzero(np.abs(delta)>3e-5)),
            full_scene_G_RMS=g_rms,scene_G_over_isolated_alpha_RMS=g_rms/delta_rms if delta_rms>1e-14 else None,
            fixed_slot_Cmax=Cmax.tolist(),fixed_slot_bound_violating_pixels=int(violations.sum()),
            fixed_slot_bound_max_excess=float(np.maximum(exceed,0).max(initial=0)),
            G_peak_where_both_isolated_alpha_zero=float(np.where((P[y0:y1,x0:x1]==0)&(Q[y0:y1,x0:x1]==0),np.abs(G).max(axis=0,initial=0),0).max(initial=0)),
            isolated_parent_visible=row['versions']['parent']['radii'][0]>0))
        print(json.dumps(dict(scene=row['scene'],event=row['event'],camera=row['camera'],
            alpha_peak=rows[-1]['isolated_alpha_peak'],violations=rows[-1]['fixed_slot_bound_violating_pixels'])),flush=True)
    summary=[]
    for scene in ['flowers','room']:
        rs=[r for r in rows if r['scene']==scene];ratios=[r['scene_G_over_isolated_alpha_RMS'] for r in rs if r['scene_G_over_isolated_alpha_RMS'] is not None]
        sums=[r['relative_discrete_sum_change'] for r in rs if r['relative_discrete_sum_change'] is not None]
        summary.append(dict(scene=scene,cases=len(rs),zero_isolated_change_cases=sum(r['isolated_alpha_RMS']==0 for r in rs),
            max_isolated_alpha_peak=max(r['isolated_alpha_peak'] for r in rs),
            median_scene_G_over_isolated_alpha_RMS=float(np.median(ratios)) if ratios else None,
            maximum_scene_G_over_isolated_alpha_RMS=max(ratios) if ratios else None,
            discrete_relative_sum_change_range=[min(sums),max(sums)] if sums else None,
            conditional_fixed_slot_bound_violation_cases=sum(r['fixed_slot_bound_violating_pixels']>0 for r in rs),
            conditional_fixed_slot_bound_violation_pixels=sum(r['fixed_slot_bound_violating_pixels'] for r in rs)))
    write_new(target,dict(all_saved_arrays_and_controls_verified=True,cases=32,rows=rows,summary=summary,
        script_sha256=sha(Path(__file__)),results_sha256=sha(OUT/'results.json'),plan_sha256=sha(OUT/'plan.json'),
        full_scene_source_audit_sha256=sha(OLD/'independent_audit_v2.json'),
        fixed_slot_bound_is_conditional_empirical=True,sorting_causally_isolated=False,
        real_GT_quality_gain=False,training=False,independent_GPU_replay=False))
    print(json.dumps(dict(all_verified=True,summary=summary)),flush=True)

if __name__=='__main__':main()
