"""Freeze new samples and training-only topology decisions before CUDA tests."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from audit_tgs_real_checkpoint import ROOT,INPUT,load_input
from tgs_topology4 import catalogue,children,world_longest_pattern,score_catalogue

OUT=ROOT/'experiments/tgs_topology_v1'
OLD=ROOT/'experiments/tgs_real_event_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'selection.json').exists(),'Frozen plan exists; inspect it, do not silently resample.'
    model,hyper,cfg,cameras,_=load_input()
    sorted_cameras=sorted(cameras,key=lambda c:c['img_name'])
    train=[{**c,'split':'train'} for i,c in enumerate(sorted_cameras) if i%8!=0]
    test=[{**c,'split':'test'} for i,c in enumerate(sorted_cameras) if i%8==0]
    old_audit=json.loads((OLD/'input_audit.json').read_text())
    old_events=json.loads((OLD/'event_selection.json').read_text())['event_ids']
    new_test=sorted(test,key=lambda c:hashlib.sha256(('tgs_real_event_v1/test/'+c['img_name']).encode()).hexdigest())[2:4]
    old_train=[c for c in old_audit['selected_cameras'] if c['split']=='train']
    eval_cameras=old_train+new_test
    assert not set(c['img_name'] for c in new_test)&set(c['img_name'] for c in old_audit['selected_cameras'])
    base=[np.load(OLD/f'baseline_{i}.npz') for i in [2,3]]
    radius=np.maximum(base[0]['radii'],base[1]['radii'])
    opacity=torch.sigmoid(model['opacity'].detach()).numpy().reshape(-1)
    eligible=np.flatnonzero((radius>24)&(opacity>=.01))
    eligible=eligible[~np.isin(eligible,old_events)]
    key=lambda i:hashlib.sha256(('tgs_topology_v1/event/'+str(int(i))).encode()).hexdigest()
    general=sorted(eligible,key=key)[:8]
    thin=((base[0]['inradius']>=1)&(base[0]['inradius']<=2.1))|((base[1]['inradius']>=1)&(base[1]['inradius']<=2.1))
    stress=[x for x in sorted(eligible[thin[eligible]],key=key) if x not in general][:8]
    ids=[('general',int(x)) for x in general]+[('stress',int(x)) for x in stress]
    patterns=catalogue();events=[]
    w,h=old_audit['actual_resolution']
    total=time.perf_counter()
    for ei,(stratum,index) in enumerate(ids):
        v=model['triangles_points'][index].detach().numpy()
        start=time.perf_counter();scores,radii,valid=score_catalogue(v.astype(float),patterns,train,w,h)
        seconds=time.perf_counter()-start
        best=float(scores.max());selected=int(np.flatnonzero(scores>=best-1e-12)[0])
        longest=world_longest_pattern(v,patterns)['id']
        options={'red4':0,'world_longest4':longest,'train_maximin4':selected}
        child_array=np.array([children(v,patterns[j]) for j in options.values()])
        np.savez(OUT/f'event_{ei}_geometry.npz',parent_id=np.array(index),parent=v,
                 children=child_array,pattern_ids=np.array(list(options.values())),scores=scores,
                 valid_train_mask=valid,training_inradii=radii)
        row={'event':ei,'parent_id':index,'stratum':stratum,'patterns':options,
             'scores':scores.tolist(),'geometry_train_view_count':int(valid.sum()),
             'geometry_train_camera_names':[c['img_name'] for c,m in zip(train,valid) if m],
             'selector_cpu_seconds':seconds}
        events.append(row)
        print(json.dumps({'event':ei,'stratum':stratum,'patterns':options,'valid_train_views':int(valid.sum()),
                          'scores_chosen':{k:float(scores[j]) for k,j in options.items()}}),flush=True)
    protocol=ROOT/'TGS_TOPOLOGY_PROTOCOL_V1.md'
    result={'frozen_before_new_split_rendering':True,'events':events,'catalogue':patterns,
            'eval_cameras':eval_cameras,'training_camera_names':[c['img_name'] for c in train],
            'training_camera_count':len(train),'width':w,'height':h,
            'general_eligible_count':len(eligible),'stress_eligible_count':int(thin[eligible].sum()),
            'general_count':len(general),'stress_count':len(stress),
            'new_test_names':[c['img_name'] for c in new_test],
            'old_event_ids_excluded':old_events,'tie_abs_tolerance':1e-12,
            'geometry_selector_uses_test_cameras':False,'new_events_disjoint_from_previous':True,
            'protocol_sha256':sha(protocol),'model_sha256':json.loads((OLD/'summary_audit.json').read_text())['input_model_sha256'],
            'geometry_planning_seconds':time.perf_counter()-total,
            'script_sha256':sha(Path(__file__)),'topology_module_sha256':sha(ROOT/'tgs_topology4.py'),
            'no_training':True}
    (OUT/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'planned_events':len(events),'maximin_equals_longest':sum(x['patterns']['train_maximin4']==x['patterns']['world_longest4'] for x in events)}),flush=True)


if __name__=='__main__':main()
