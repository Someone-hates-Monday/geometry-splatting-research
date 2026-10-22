"""CPU evidence audit of saved real-checkpoint split simulations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from audit_tgs_real_checkpoint import ROOT,INPUT,OUT


def main():
    result=json.loads((OUT/'results.json').read_text());assert result['complete'] and result['no_training']
    audit=json.loads((OUT/'input_audit.json').read_text())
    selection=json.loads((OUT/'event_selection.json').read_text())
    assert hashlib.sha256((OUT/'input_audit.json').read_bytes()).hexdigest()==selection['source_audit_sha256']
    eligible=np.load(OUT/'eligible_ids.npy')
    assert hashlib.sha256(eligible.astype('<i8').tobytes()).hexdigest()==selection['eligible_ids_sha256']
    key=lambda i:hashlib.sha256(('tgs_real_event_v1/event/'+str(int(i))).encode()).hexdigest()
    assert [int(x) for x in sorted(eligible,key=key)[:8]]==selection['event_ids']
    baselines=[np.load(OUT/f'baseline_{i}.npz') for i in range(4)]
    max_radius=np.maximum.reduce([b['radii'] for b in baselines])
    chosen=json.loads((INPUT/'selection.json').read_text())
    modelpath=INPUT/chosen['scene']/'point_cloud'/chosen['iteration']/'point_cloud_state_dict.pt'
    receipt=json.loads(modelpath.with_suffix('.pt.receipt.json').read_text())
    assert hashlib.sha256(modelpath.read_bytes()).hexdigest()==receipt['sha256']
    model=torch.load(modelpath,map_location='cpu',weights_only=True)
    opacity=torch.sigmoid(model['opacity'].detach()).numpy().reshape(-1)
    expected=np.flatnonzero((max_radius>24)&(opacity>=.01))
    assert np.array_equal(expected,eligible)
    for ei,index in enumerate(selection['event_ids']):
        raw=np.load(OUT/f'event_{ei}_input.npz')
        assert np.array_equal(model['triangles_points'][index].detach().numpy(),raw['points'])
        sh=torch.cat([model['features_dc'][index],model['features_rest'][index]],0).detach().numpy()
        assert np.array_equal(sh,raw['sh'])
        a,b,c=raw['points'];ab,ac,bc=(a+b)/2,(a+c)/2,(b+c)/2
        children=np.array([[a,ab,ac],[b,ab,bc],[c,ac,bc],[ab,ac,bc]])
        assert np.array_equal(children,raw['children'])
        sigma=float((.01+torch.exp(model['sigma'][index].detach())).item())
        assert abs(sigma-float(raw['sigma']))<1e-6 and abs(opacity[index]-float(raw['opacity']))<1e-7
    assert len(result['cases'])==32
    h,w=audit['actual_resolution'][1],audit['actual_resolution'][0]
    verified=0;mode_stats={};continuous_visible=[];continuous_all=[]
    for row in result['cases']:
        full_alpha=[]
        for mode in ['original','fixed','white']:
            path=OUT/f"event_{row['event_index']}_camera_{row['camera']}_{mode}.npz"
            raw=np.load(path);x0,y0,x1,y1=raw['bbox'];b=baselines[row['camera']]
            assert raw['shape'].tolist()==[h,w]
            assert 0<=x0<=x1<=w and 0<=y0<=y1<=h
            if mode!='white':
                assert np.array_equal(raw['before_rgb'],b[mode+'_rgb'][:,y0:y1,x0:x1])
            else:
                assert np.max(np.abs(raw['before_rgb']-b['white_alpha'][None,y0:y1,x0:x1]),initial=0)<3e-5
            assert np.array_equal(raw['before_alpha'],b['white_alpha'][y0:y1,x0:x1])
            assert np.array_equal(raw['before_depth'],b['depth'][y0:y1,x0:x1])
            d=np.abs(raw['after_rgb'].astype(float)-raw['before_rgb'])
            da=np.abs(raw['after_alpha'].astype(float)-raw['before_alpha'])
            dd=np.abs(raw['after_depth'].astype(float)-raw['before_depth'])
            metrics={'rgb_max_abs':float(d.max(initial=0)),'rgb_L1_sum':float(d.sum()),
                     'rgb_mean_abs':float(d.sum()/(3*h*w)),
                     'rgb_pixels_above_tolerance':int((d.max(0,initial=0)>selection['rgb_tolerance']).sum()),
                     'rgb_pixels_nonzero':int((d.max(0,initial=0)>0).sum()),
                     'alpha_L1_sum':float(da.sum()),'alpha_max_abs':float(da.max(initial=0)),
                     'depth_max_abs':float(dd.max(initial=0))}
            assert all(abs(v-row['metrics'][mode][k])<1e-11 for k,v in metrics.items())
            full=b['white_alpha'].copy();full[y0:y1,x0:x1]=raw['after_alpha'];full_alpha.append(full)
            verified+=1
        assert all(np.array_equal(full_alpha[0],x) for x in full_alpha[1:])
        pa=row['continuous_parent'];ch=row['continuous_children']
        if pa is not None and all(x is not None for x in ch):
            mass=sum(x['mass'] for x in ch)
            mu=sum(x['mass']*np.array(x['centroid']) for x in ch)/mass
            item={'event':row['event_index'],'camera':row['camera'],
                  'displacement':float(np.linalg.norm(mu-np.array(pa['centroid']))),
                  'mass_relative_difference':mass/pa['mass']-1}
            continuous_all.append(item)
            if row['parent_radii']>0:continuous_visible.append(item)
    for mode in ['original','fixed','white']:
        values=[x['metrics'][mode] for x in result['cases']]
        mode_stats[mode]={'cases_with_above_tolerance_pixels':sum(x['rgb_pixels_above_tolerance']>0 for x in values),
                         'max_RGB_abs':max(x['rgb_max_abs'] for x in values),
                         'max_full_image_MAE':max(x['rgb_mean_abs'] for x in values),
                         'sum_case_pixel_counts':sum(x['rgb_pixels_above_tolerance'] for x in values),
                         'median_case_peak':float(np.median([x['rgb_max_abs'] for x in values]))}
    cull=[{'event':x['event_index'],'camera':x['camera'],'parent_id':x['parent_id'],
           'parent_inradius':x['parent_inradius'],'rgb_peak':x['metrics']['original']['rgb_max_abs']}
          for x in result['cases'] if x['parent_radii']>0 and max(x['child_radii'])==0]
    summary={'audit_passed':True,'saved_variants_checked':verified,'input_model_sha256':receipt['sha256'],
             'candidate_count':len(eligible),'event_count':8,'camera_count':4,'event_camera_cases':32,
             'parent_survives_preprocessing':sum(x['parent_radii']>0 for x in result['cases']),
             'mode_statistics':mode_stats,'surviving_parent_all_children_culled':cull,
             'continuous_visible_parent_cases':continuous_visible,
             'continuous_other_cases_count':len(continuous_all)-len(continuous_visible),
             'continuous_nonpositive_depth_cases':32-len(continuous_all),
             'baseline_SH_checks':selection['baseline_SH_checks'],
             'no_training_benefit_claim':True,'event_label':audit['event_label'],
             'audit_scope':'input tensors, deterministic selection, 96 saved crop metrics and identical alpha across appearance modes; no independent GPU replay'}
    (OUT/'summary_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['continuous_visible_parent_cases','baseline_SH_checks']},indent=2))


if __name__=='__main__':
    main()
