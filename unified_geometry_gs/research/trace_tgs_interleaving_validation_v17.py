"""12 fixed native forwards; 256 tiny observer calls; no training or new build."""
import argparse,gc,hashlib,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,module,pack,split,CrossWorkload,exact
from tgs_sh_isolation_common_v12 import capture
from trace_tgs_window_scene_v15 import call,gather
OUT=ROOT/'experiments/tgs_interleaving_v17/validation';OLD=ROOT/'experiments/tgs_split_color_v14'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);old=json.loads((OLD/'plan.json').read_text());results=json.loads((OLD/'results.json').read_text());cases=[]
    files=[Path(__file__),ROOT/'TGS_INTERLEAVING_VALIDATION_PROTOCOL_V17.md',ROOT/'TGS_INTERLEAVING_CERTIFICATE_NOTE_V17.md',
        ROOT/'tgs_interleaving_certificate_v17.py',ROOT/'trace_tgs_window_scene_v15.py',ROOT/'tgs_window_scene_trace_v15.cu',
        ROOT/'tgs_split_color_common_v14.py',ROOT/'tgs_sh_isolation_common_v12.py',ROOT/'run_tgs_real_events.py',ROOT/'tgs_cross_scene_common_v10.py',
        OLD/'plan.json',OLD/'results.json',ROOT/'experiments/tgs_window_scene_v15/trace/build_receipt.json']
    for scene in ['flowers','room']:
        sc=next(s for s in old['scenes'] if s['scene']==scene);W,H=sc['resolution'];files.extend(ROOT/m['path'] for m in old['models'] if m['scene']==scene)
        for ci in [0,1]:
            files.extend([OLD/scene/f'baseline_{ci}'/n for n in ['receipt.json','colors.npy','camera.npz']])
            for event in [1,2]:
                r=next(x for x in results['cases'] if x['scene']==scene and x['camera']==ci and x['event']==event);x0,y0,x1,y1=r['bbox'];pts=[]
                if x1>x0 and y1>y0:
                    pts=[[min(x1-1,x0+(x1-x0)*i//4),min(y1-1,y0+(y1-y0)*j//4)] for j in [1,3] for i in [1,3]]
                k=0
                while len(pts)<8:
                    h=hashlib.sha256(f'v17/{scene}/{ci}/{event}/{k}'.encode()).digest();p=[int.from_bytes(h[:8],'little')%W,int.from_bytes(h[8:16],'little')%H];k+=1
                    if p not in pts:pts.append(p)
                assert len(set(map(tuple,pts)))==8
                cases.append(dict(scene=scene,camera=ci,event=event,parent_id=r['parent_id'],pixels=pts,bbox=r['bbox'],expected=r['hold_hashes']))
                files.extend([OLD/scene/f'event_{event}'/'input.npz',OLD/scene/f'event_{event}'/f'camera_{ci}'/'receipt.json'])
    obs=json.loads((ROOT/'experiments/tgs_window_scene_v15/trace/build_receipt.json').read_text());assert sha(Path(obs['path']))==obs['sha256']
    native=old['modules']['native'];assert sha(Path(native['path']))==native['sha256']
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},cases=cases,native=native,observer=obs,
        full_forward_cap=12,trace_call_cap=256,training=False,independent_photographic_validation=False))
    print('VALIDATION_PLAN_FROZEN',flush=True)

@torch.no_grad()
def run():
    plan=json.loads((OUT/'plan.json').read_text());assert not (OUT/'results.json').exists()
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    for key in ['native','observer']:
        en=plan[key];assert sha(Path(en['path']))==en.get('sha256')
    native=module(plan['native']['path']);observer=module(plan['observer']['path']);forwards=traces=0;rows=[]
    def save_endpoint(res,state,cam,col,case,point,mode):
        nonlocal traces
        x,y=point;packet,ids,endpoint,meta=gather(res,state,cam,col,x,y)
        p=torch.from_numpy(packet.copy()).cuda();actual,summary=observer.replay(p,x,y,False);alltrace,fullsummary=observer.replay(p,x,y,True);torch.cuda.synchronize();traces+=2
        actual=actual.cpu().numpy();summary=summary.cpu().numpy();alltrace=alltrace.cpu().numpy()
        assert exact(summary[:5],endpoint.astype(np.float32)),(case,point,mode)
        name=f"{case['scene']}_c{case['camera']}_e{case['event']}_{x}_{y}_{mode}.npz"
        np.savez_compressed(OUT/name,packed=packet,actual=actual,summary=summary,full=alltrace,fullsummary=fullsummary.cpu().numpy(),native=endpoint,
            parent_color=col[case['parent_id'] if mode=='before' else len(state['points'])-1].cpu().numpy())
        return name
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh);N=len(w.points)
        for ci,cam in enumerate(w.cameras):
            cases=[x for x in plan['cases'] if x['scene']==scene and x['camera']==ci];col=torch.from_numpy(np.load(OLD/scene/f'baseline_{ci}/colors.npy')).cuda()
            res=call(native,state,cam,col);torch.cuda.synchronize();forwards+=1;arrays=capture(res,cam['w'],cam['h'])
            expected=json.loads((OLD/scene/f'baseline_{ci}/receipt.json').read_text())['arrays']
            for k,a in arrays.items():assert ahash(a)==expected[k]['sha256'],(scene,ci,k)
            old_names={}
            for case in cases:
                for point in case['pixels']:old_names[(case['event'],*point)]=save_endpoint(res,state,cam,col,case,point,'before')
            del arrays,res
            for case in cases:
                index=case['parent_id'];changed=split(state,index);cc=torch.cat([col[:index],col[index+1:],col[index:index+1].repeat(4,1)]).contiguous()
                res=call(native,changed,cam,cc);torch.cuda.synchronize();forwards+=1;arrays=capture(res,cam['w'],cam['h'])
                for k,a in arrays.items():assert ahash(a)==case['expected'][k]['sha256'],(scene,ci,index,k)
                for point in case['pixels']:
                    after=save_endpoint(res,changed,cam,cc,case,point,'after')
                    rows.append(dict(scene=scene,camera=ci,event=case['event'],parent_id=index,N=N,pixel=point,before=old_names[(case['event'],*point)],after=after))
                del res,arrays,changed,cc
                print(json.dumps(dict(scene=scene,camera=ci,event=case['event'],full_forwards=forwards,trace_calls=traces)),flush=True)
        assert w.input_hashes()==initial
        del w,state,col;gc.collect();torch.cuda.empty_cache()
    assert forwards==12 and traces==256 and len(rows)==64
    write_new(OUT/'results.json',dict(rows=rows,full_forwards=forwards,trace_calls=traces,native_full_fields_literal=True,
        all_endpoints_literal=True,plan_sha256=sha(OUT/'plan.json'),training=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
