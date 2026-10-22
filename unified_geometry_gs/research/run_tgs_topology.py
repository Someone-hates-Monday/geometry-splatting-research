"""Frozen equal-capacity topology comparison; no training or optimizer."""
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from audit_tgs_real_checkpoint import ROOT,load_input
from tgs_native_diagnostic_backend import load_backend
from run_tgs_real_events import pack,render,camera_tensors,metrics,save_change

OUT=ROOT/'experiments/tgs_topology_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def replace_parent(state,index,child):
    attrs={};n=len(state['points'])
    child=torch.tensor(child,device='cuda',dtype=torch.float32)
    for name in ['points','sigma','opacity','sh']:
        v=state[name];tail=child if name=='points' else v[index:index+1].repeat((4,)+(1,)*(v.ndim-1))
        attrs[name]=torch.cat([v[:index],v[index+1:],tail])
        assert len(attrs[name])==n+3 and torch.equal(attrs[name][:index],v[:index])
        assert torch.equal(attrs[name][index:n-1],v[index+1:])
        if name!='points':assert torch.equal(attrs[name][-4:],v[index:index+1].expand_as(tail))
    return pack(attrs['points'],attrs['sigma'],attrs['opacity'],attrs['sh'])


def main():
    started=time.time()
    assert not (OUT/'results.json').exists(),'Finished run exists; audit it instead of restarting.'
    plan=json.loads((OUT/'selection.json').read_text())
    assert plan['frozen_before_new_split_rendering'] and plan['protocol_sha256']==sha(ROOT/'TGS_TOPOLOGY_PROTOCOL_V1.md')
    selection_hash=sha(OUT/'selection.json')
    model,hyper,cfg,all_cameras,selection=load_input()
    model={k:(v.detach().cuda() if isinstance(v,torch.Tensor) else v) for k,v in model.items()}
    base=pack(model['triangles_points'],.01+torch.exp(model['sigma']),torch.sigmoid(model['opacity']),
              torch.cat([model['features_dc'],model['features_rest']],1))
    mod=load_backend(shfix=True)
    cams=[camera_tensors(c,plan['width'],plan['height']) for c in plan['eval_cameras']]
    baselines=[];repeat=0.
    for ci,cam in enumerate(cams):
        color=render(mod,base,cam)
        white=render(mod,base,cam,'white',vector_ids=slice(0,0))
        assert np.array_equal(color['alpha'],white['alpha'])
        if ci==0:
            for _ in range(2):
                again=render(mod,base,cam,vector_ids=slice(0,0))
                repeat=max(repeat,float(np.abs(color['rgb']-again['rgb']).max()))
        baselines.append({'sh':color,'white':white})
        np.savez_compressed(OUT/f'baseline_{ci}.npz',fixed_rgb=color['rgb'],white_rgb=white['rgb'],
            alpha=color['alpha'],depth=color['depth'],radii=color['radii'],inradius=color['inradius'],
            max_blend=color['max_blend'],view=cam['view'].cpu().numpy(),projection=cam['projection'].cpu().numpy())
        print(json.dumps({'stage':'baseline','camera':ci,'split':plan['eval_cameras'][ci]['split']}),flush=True)
    tolerance=max(3e-5,100*repeat)
    (OUT/'calibration.json').write_text(json.dumps({'repeat_max':repeat,'tolerance':tolerance,
                'selection_sha256':selection_hash,'saved_before_new_split_rendering':True},indent=2)+'\n')
    rows=[];native_split_calls=0
    for event in plan['events']:
        ei,index=event['event'],event['parent_id']
        geometry=np.load(OUT/f'event_{ei}_geometry.npz')
        assert np.array_equal(geometry['parent'],base['points'][index].cpu().numpy())
        np.savez(OUT/f'event_{ei}_attributes.npz',parent_id=np.array(index),sigma=base['sigma'][index].cpu().numpy(),
                 opacity=base['opacity'][index].cpu().numpy(),sh=base['sh'][index].cpu().numpy(),
                 raw_sigma=model['sigma'][index].cpu().numpy(),raw_opacity=model['opacity'][index].cpu().numpy())
        # Exact same input geometry can share a recorded renderer call. These
        # aliases are labelled, not counted as independent GPU executions.
        cache={}
        for si,style in enumerate(['red4','world_longest4','train_maximin4']):
            child=geometry['children'][si]
            identity=hashlib.sha256(child.tobytes()).hexdigest()
            if identity in cache:
                for old in cache[identity]:
                    rows.append({**old,'style':style,'aliased_to_style':old['style']})
                print(json.dumps({'stage':'alias','event':ei,'style':style,'same_geometry_as':cache[identity][0]['style']}),flush=True)
                continue
            state=replace_parent(base,index,child);n=len(state['points']);new_rows=[]
            for ci,cam in enumerate(cams):
                row={'event':ei,'parent_id':index,'stratum':event['stratum'],'style':style,'camera':ci,
                     'split':plan['eval_cameras'][ci]['split'],'N_after':n,'metrics':{},'raw_files':{},
                     'parent_radii':int(baselines[ci]['sh']['radii'][index]),
                     'parent_inradius':float(baselines[ci]['sh']['inradius'][index])}
                alpha=None
                for appearance in ['sh','white']:
                    after=render(mod,state,cam,appearance,vector_ids=slice(n-4,n));native_split_calls+=1
                    before=baselines[ci][appearance]
                    row['metrics'][appearance]=metrics(before,after,tolerance)
                    file=OUT/f'event_{ei}_camera_{ci}_{style}_{appearance}.npz'
                    save_change(file,before,after);row['raw_files'][appearance]=file.name
                    if appearance=='sh':
                        alpha=after['alpha']
                        row.update({'child_radii':after['radii'].tolist(),'child_inradius':after['inradius'].tolist(),
                                    'child_max_blend':after['max_blend'].tolist()})
                    else:assert np.array_equal(alpha,after['alpha'])
                rows.append(row);new_rows.append(row)
                print(json.dumps({'stage':'split','event':ei,'stratum':event['stratum'],'style':style,'camera':ci,
                    'children_survive':sum(x>0 for x in row['child_radii']),
                    'RGB_peak':row['metrics']['sh']['rgb_max_abs']}),flush=True)
            cache[identity]=new_rows
            del state
        (OUT/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
    report={'complete':True,'N_before':len(base['points']),'cases':rows,'selection_sha256':selection_hash,
            'protocol_sha256':plan['protocol_sha256'],'native_module_sha256':sha(Path(mod.__file__)),
            'native_split_render_calls':native_split_calls,'alias_case_count':sum('aliased_to_style' in x for x in rows),
            'mode_count':2,'no_training':True,'seconds':time.time()-started,'rgb_tolerance':tolerance,
            'scope':'one real checkpoint, fixed four-child capacity, frozen prospective cases; no quality/optimization benefit demonstrated'}
    assert len(rows)==len(plan['events'])*4*3
    (OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':True,'cases':len(rows),'native_split_calls':native_split_calls,'seconds':report['seconds']}),flush=True)


if __name__=='__main__':
    try:
        with torch.no_grad():main()
    except Exception:
        (OUT/'failure.txt').write_text(traceback.format_exc());raise
