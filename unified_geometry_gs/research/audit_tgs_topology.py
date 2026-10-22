"""Recompute frozen topology choices and every saved metric on CPU.

Audits records; does not replay CUDA or imply training benefit.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from audit_tgs_real_checkpoint import ROOT,load_input
from tgs_topology4 import catalogue,children,world_longest_pattern,score_catalogue

OUT=ROOT/'experiments/tgs_topology_v1'
OLD=ROOT/'experiments/tgs_real_event_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(pattern):
    return tuple(sorted(tuple(sorted(tuple(v) for v in tri)) for tri in pattern['barycentric_quarters']))


def main():
    plan=json.loads((OUT/'selection.json').read_text());result=json.loads((OUT/'results.json').read_text())
    assert result['complete'] and result['no_training'] and result['selection_sha256']==sha(OUT/'selection.json')
    assert plan['protocol_sha256']==result['protocol_sha256']==sha(ROOT/'TGS_TOPOLOGY_PROTOCOL_V1.md')
    assert plan['script_sha256']==sha(ROOT/'plan_tgs_topology.py') and plan['topology_module_sha256']==sha(ROOT/'tgs_topology4.py')
    patterns=catalogue();assert patterns==plan['catalogue'] and len(patterns)==28
    model,hyper,cfg,cameras,_=load_input();n=len(model['triangles_points'])
    camera_sorted=sorted(cameras,key=lambda c:c['img_name'])
    train=[c for i,c in enumerate(camera_sorted) if i%8!=0]
    test=[c for i,c in enumerate(camera_sorted) if i%8==0]
    assert [c['img_name'] for c in train]==plan['training_camera_names']
    order=sorted(test,key=lambda c:hashlib.sha256(('tgs_real_event_v1/test/'+c['img_name']).encode()).hexdigest())
    assert [x['img_name'] for x in order[2:4]]==plan['new_test_names']
    old_audit=json.loads((OLD/'input_audit.json').read_text())
    old_ids=json.loads((OLD/'event_selection.json').read_text())['event_ids']
    b=[np.load(OLD/f'baseline_{i}.npz') for i in [2,3]]
    radius=np.maximum(b[0]['radii'],b[1]['radii']);op=torch.sigmoid(model['opacity'].detach()).numpy().reshape(-1)
    eligible=np.flatnonzero((radius>24)&(op>=.01));eligible=eligible[~np.isin(eligible,old_ids)]
    key=lambda i:hashlib.sha256(('tgs_topology_v1/event/'+str(int(i))).encode()).hexdigest()
    general=sorted(eligible,key=key)[:8]
    thin=((b[0]['inradius']>=1)&(b[0]['inradius']<=2.1))|((b[1]['inradius']>=1)&(b[1]['inradius']<=2.1))
    stress=[i for i in sorted(eligible[thin[eligible]],key=key) if i not in general][:8]
    assert [e['parent_id'] for e in plan['events']]==list(map(int,general+stress))
    geometries={};max_world_area_error=0.
    for event in plan['events']:
        ei,index=event['event'],event['parent_id'];raw=np.load(OUT/f'event_{ei}_geometry.npz')
        v=model['triangles_points'][index].detach().numpy();assert np.array_equal(v,raw['parent'])
        scores,radii,valid=score_catalogue(v.astype(float),patterns,train,plan['width'],plan['height'])
        assert np.array_equal(valid,raw['valid_train_mask']) and np.array_equal(radii,raw['training_inradii'])
        assert np.array_equal(scores,raw['scores']) and np.array_equal(scores,np.array(event['scores']))
        best=int(np.flatnonzero(scores>=scores.max()-1e-12)[0])
        assert best==event['patterns']['train_maximin4']
        assert world_longest_pattern(v,patterns)['id']==event['patterns']['world_longest4']
        for si,style in enumerate(['red4','world_longest4','train_maximin4']):
            c=children(v,patterns[event['patterns'][style]])
            assert np.array_equal(c,raw['children'][si]) and c.shape==(4,3,3)
            # Quantized world coordinates can differ from ideal dyadic barycentric
            # geometry. Report the actual error, without silently treating it as zero.
            vf=v.astype(float);cf=c.astype(float)
            parent_area=np.linalg.norm(np.cross(vf[1]-vf[0],vf[2]-vf[0]))
            ratios=np.linalg.norm(np.cross(cf[:,1]-cf[:,0],cf[:,2]-cf[:,0]),axis=1)/parent_area
            max_world_area_error=max(max_world_area_error,float(np.max(np.abs(ratios-.25))))
        attr=np.load(OUT/f'event_{ei}_attributes.npz')
        assert np.array_equal(attr['raw_sigma'],model['sigma'][index].detach().numpy())
        assert np.array_equal(attr['raw_opacity'],model['opacity'][index].detach().numpy())
        assert np.array_equal(attr['sh'],torch.cat([model['features_dc'][index],model['features_rest'][index]],0).detach().numpy())
        geometries[ei]=raw
    baselines=[np.load(OUT/f'baseline_{i}.npz') for i in range(4)]
    checked=0;unique_files=set();h,w=plan['height'],plan['width']
    for row in result['cases']:
        assert row['N_after']==n+3
        b=baselines[row['camera']];full_alphas=[]
        for mode in ['sh','white']:
            p=OUT/row['raw_files'][mode];unique_files.add(p.name);raw=np.load(p)
            x0,y0,x1,y1=raw['bbox'];assert raw['shape'].tolist()==[h,w]
            assert np.array_equal(raw['before_rgb'],b['fixed_rgb' if mode=='sh' else 'white_rgb'][:,y0:y1,x0:x1])
            assert np.array_equal(raw['before_alpha'],b['alpha'][y0:y1,x0:x1])
            assert np.array_equal(raw['before_depth'],b['depth'][y0:y1,x0:x1])
            d=np.abs(raw['after_rgb'].astype(float)-raw['before_rgb']);da=np.abs(raw['after_alpha'].astype(float)-raw['before_alpha'])
            dd=np.abs(raw['after_depth'].astype(float)-raw['before_depth'])
            metrics={'rgb_max_abs':float(d.max(initial=0)),'rgb_L1_sum':float(d.sum()),
                     'rgb_mean_abs':float(d.sum()/(3*h*w)),
                     'rgb_pixels_above_tolerance':int((d.max(0,initial=0)>result['rgb_tolerance']).sum()),
                     'rgb_pixels_nonzero':int((d.max(0,initial=0)>0).sum()),'alpha_L1_sum':float(da.sum()),
                     'depth_max_abs':float(dd.max(initial=0)),'alpha_max_abs':float(da.max(initial=0))}
            assert all(abs(x-row['metrics'][mode][k])<1e-11 for k,x in metrics.items())
            alpha=b['alpha'].copy();alpha[y0:y1,x0:x1]=raw['after_alpha'];full_alphas.append(alpha)
            checked+=1
        assert np.array_equal(full_alphas[0],full_alphas[1])
    assert len(result['cases'])==192 and checked==384
    assert len(unique_files)==result['native_split_render_calls']==272
    stats=[]
    for stratum in ['general','stress']:
        for split in ['train','test']:
            for style in ['red4','world_longest4','train_maximin4']:
                rows=[r for r in result['cases'] if (r['stratum'],r['split'],r['style'])==(stratum,split,style)]
                visible=[r for r in rows if r['parent_radii']>0]
                stats.append({'stratum':stratum,'split':split,'style':style,'cases':len(rows),
                    'parent_preprocess_survives':len(visible),
                    'all_children_culled':sum(max(r['child_radii'])==0 for r in visible),
                    'partial_child_cull':sum(0<sum(x>0 for x in r['child_radii'])<4 for r in visible),
                    'surviving_child_count':sum(sum(x>0 for x in r['child_radii']) for r in visible),
                    'median_SH_full_image_MAE':float(np.median([r['metrics']['sh']['rgb_mean_abs'] for r in rows])),
                    'sum_SH_RGB_L1':sum(r['metrics']['sh']['rgb_L1_sum'] for r in rows)})
    gates=[]
    for stratum in ['general','stress']:
        long=next(x for x in stats if (x['stratum'],x['split'],x['style'])==(stratum,'test','world_longest4'))
        selected=next(x for x in stats if (x['stratum'],x['split'],x['style'])==(stratum,'test','train_maximin4'))
        informative=long['all_children_culled']>0 or selected['all_children_culled']>0
        fewer=selected['all_children_culled']<long['all_children_culled']
        mae=selected['median_SH_full_image_MAE']<=long['median_SH_full_image_MAE']
        gates.append({'stratum':stratum,'attrition_informative':informative,'strictly_fewer_test_all_culls':fewer,
                      'test_median_MAE_not_worse':mae,'gate_pass':informative and fewer and mae})
    c=plan['events'];output={'passed':True,'saved_conditions_checked':checked,'unique_native_files':len(unique_files),
        'alias_case_count':result['alias_case_count'],'capacity_N_to_Nplus3_verified':True,
        'ideal_equal_area_partition':'exact two-level bisection construction; all four barycentric area ratios 1/4',
        'actual_float32_max_abs_child_area_ratio_error':max_world_area_error,
        'selector_red4_event_count':sum(x['patterns']['train_maximin4']==0 for x in c),
        'selector_physically_equal_to_longest_count':sum(canonical(patterns[x['patterns']['train_maximin4']])==canonical(patterns[x['patterns']['world_longest4']]) for x in c),
        'stats':stats,'gates':gates,'advance_selector_as_training_candidate':all(g['gate_pass'] for g in gates),
        'training_or_quality_benefit_claim':False,'scope':'CPU record/geometry audit, not independent GPU replay'}
    (OUT/'summary_audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='stats'},indent=2),flush=True)


if __name__=='__main__':main()
