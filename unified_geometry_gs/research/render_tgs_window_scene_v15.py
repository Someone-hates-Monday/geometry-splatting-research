"""64 tiny native forwards on saved real-model geometries/cameras; no full-scene rerender."""
import argparse,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,module,pack,render,exact
OLD=ROOT/'experiments/tgs_split_color_v14'
OUT=ROOT/'experiments/tgs_window_scene_v15'
PROTOCOL=ROOT/'TGS_WINDOW_SCENE_PROTOCOL_V15.md'
PRIOR='8b7736cf7e650cc960a23c19d3660f1db0fde4ab51480321404465ee9aed5411'

def prepare():
    OUT.mkdir(exist_ok=True);assert not (OUT/'plan.json').exists()
    assert sha(OLD/'evidence_manifest.json')==PRIOR
    oldplan=json.loads((OLD/'plan.json').read_text());res=json.loads((OLD/'results.json').read_text())
    assert json.loads((OLD/'independent_audit_v2.json').read_text())['all_verified']
    inputs={n:sha(OLD/n) for n in ['evidence_manifest.json','plan.json','results.json','independent_audit_v2.json']}
    for r in res['cases']:
        base=OLD/r['scene'];folder=base/f"event_{r['event']}"/f"camera_{r['camera']}"
        for p in [base/f"event_{r['event']}/input.npz",folder/'changes.npz',base/f"baseline_{r['camera']}/camera.npz",base/f"baseline_{r['camera']}/colors.npy"]:
            inputs[str(p.relative_to(OLD))]=sha(p)
    scripts=['render_tgs_window_scene_v15.py','audit_tgs_window_scene_v15.py','tgs_split_color_common_v14.py','run_tgs_real_events.py']
    en=oldplan['modules']['native'];assert sha(Path(en['path']))==en['sha256']
    write_new(OUT/'plan.json',dict(protocol_sha256=sha(PROTOCOL),old_inputs=inputs,module=en,
        scripts={n:sha(ROOT/n) for n in scripts},cases=[{k:r[k] for k in ['scene','event','camera','parent_id','parent_radii','child_radii']} for r in res['cases']],
        forward_cap=64,posthoc_same_cases=True,training=False,benchmark=False))
    print(json.dumps(dict(plan_frozen=True,cases=len(res['cases']))),flush=True)

def crop_save(path,arrays):
    assert exact(arrays['rgb'][0],arrays['rgb'][1]) and exact(arrays['rgb'][1],arrays['rgb'][2])
    alpha=arrays['aux'][1];rgb=arrays['rgb'][0]
    error=float(np.abs(alpha.astype(float)-rgb.astype(float)).max());assert error<=2e-6
    mask=(alpha!=0)|(rgb!=0);yy,xx=np.nonzero(mask)
    box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else [0,0,0,0]
    x0,y0,x1,y1=box
    np.savez_compressed(path,bbox=np.array(box),shape=np.array(alpha.shape),alpha=alpha[y0:y1,x0:x1],
        rgb=rgb[y0:y1,x0:x1],radii=arrays['radii'])
    for name,value in [('alpha',alpha),('rgb',rgb)]:
        restored=np.zeros_like(value);restored[y0:y1,x0:x1]=value[y0:y1,x0:x1];assert exact(restored,value)
    return dict(bbox=box,shape=list(alpha.shape),alpha_sha256=ahash(alpha),rgb_sha256=ahash(rgb),
                dtype=str(alpha.dtype),white_alpha_maxabs=error,radii=arrays['radii'].tolist(),file_sha256=sha(path))

@torch.no_grad()
def run():
    plan=json.loads((OUT/'plan.json').read_text());assert not (OUT/'results.json').exists()
    assert sha(PROTOCOL)==plan['protocol_sha256']
    for n,h in plan['scripts'].items():assert sha(ROOT/n)==h
    for n,h in plan['old_inputs'].items():assert sha(OLD/n)==h
    en=plan['module'];assert sha(Path(en['path']))==en['sha256'];mod=module(en['path'])
    oldplan=json.loads((OLD/'plan.json').read_text());rows=[];count=0
    for case in plan['cases']:
        base=OLD/case['scene'];raw=np.load(base/f"event_{case['event']}/input.npz")
        camera=np.load(base/f"baseline_{case['camera']}/camera.npz")
        sc=next(s for s in oldplan['scenes'] if s['scene']==case['scene']);meta=sc['cameras'][case['camera']]
        width,height=sc['resolution'];cam={k:torch.from_numpy(camera[k].copy()).cuda() for k in ['view','projection','center']}
        cam.update(w=width,h=height,tx=meta['width']/(2*meta['fx']),ty=meta['height']/(2*meta['fy']))
        folder=OUT/case['scene']/f"event_{case['event']}"/f"camera_{case['camera']}";folder.mkdir(parents=True)
        versions={}
        for name,points in [('parent',raw['parent_points'][None]),('children',raw['child_points'])]:
            n=len(points);state=pack(torch.tensor(points,device='cuda'),torch.tensor(raw['sigma'],device='cuda').repeat(n,1),
                torch.tensor(raw['opacity'],device='cuda').repeat(n,1),torch.zeros((n,16,3),device='cuda'))
            arrays,tiles,_=render(mod,state,cam,torch.ones((n,3),device='cuda'));count+=1
            expected=[case['parent_radii']] if name=='parent' else case['child_radii']
            assert arrays['radii'].tolist()==expected
            record=crop_save(folder/(name+'.npz'),arrays);record['tiles']=tiles;versions[name]=record
            del state,arrays
        row=dict(**case,versions=versions);write_new(folder/'receipt.json',row);rows.append(row)
        (OUT/'progress.json').write_text(json.dumps(dict(forwards=count,cases=rows),indent=2)+'\n')
        print(json.dumps(dict(scene=case['scene'],event=case['event'],camera=case['camera'],forwards=count)),flush=True)
    assert count==64 and len(rows)==32
    write_new(OUT/'results.json',dict(cases=rows,forwards=count,plan_sha256=sha(OUT/'plan.json'),
        script_sha256=sha(Path(__file__)),all_radii_and_white_controls=True,training=False,benchmark=False))
    print(json.dumps(dict(complete=True,forwards=count,cases=len(rows))),flush=True)

if __name__=='__main__':
    try:
        p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args()
        prepare() if args.action=='prepare' else run()
    except BaseException:
        OUT.mkdir(exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());raise
