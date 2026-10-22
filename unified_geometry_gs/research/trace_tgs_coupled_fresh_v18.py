"""Separate-camera6-forward fixed-input diagnostic; no build/optimizer/backward."""
import argparse,gc,hashlib,json,math,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,ahash,write_new,module,pack,split,colors,exact
from tgs_sh_isolation_common_v12 import capture,workload
from trace_tgs_window_scene_v15 import call,gather
OUT=ROOT/'experiments/tgs_coupled_v18/fresh';OLD=ROOT/'experiments/tgs_split_color_v14'
V12=ROOT/'experiments/tgs_sh_isolation_v12'

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);old=json.loads((OLD/'plan.json').read_text());camplan=json.loads((V12/'input_plan.json').read_text());identity=json.loads((V12/'identity_audit.json').read_text());cases=[]
    files=[Path(__file__),ROOT/'TGS_COUPLED_FRESH_PROTOCOL_V18.md',ROOT/'TGS_COUPLED_CERTIFICATE_NOTE_V18.md',ROOT/'tgs_coupled_certificate_v18.py',
        ROOT/'tgs_interleaving_certificate_v17.py',ROOT/'audit_tgs_interleaving_validation_v17_v2.py',ROOT/'audit_tgs_compensated_depth_trace.py',
        ROOT/'tgs_split_color_common_v14.py',ROOT/'tgs_sh_isolation_common_v12.py',ROOT/'trace_tgs_window_scene_v15.py',ROOT/'tgs_window_scene_trace_v15.cu',
        ROOT/'run_tgs_real_events.py',ROOT/'tgs_cross_scene_common_v10.py',HELPER,OLD/'plan.json',V12/'input_plan.json',V12/'identity_audit.json',
        ROOT/'experiments/tgs_window_scene_v15/trace/build_receipt.json',OUT.parent/'development_plan.json',OUT.parent/'development_audit.json']
    for scene in ['flowers','room']:
        r=next(x for x in camplan['rows'] if x['scene']==scene and x['cohort']=='fresh');c=r['camera']
        ident=next(x for x in identity['rows'] if x['scene']==scene and x['cohort']=='fresh');H,W=ident['shape'][:2]
        c2w=np.eye(4);c2w[:3,:3]=c['rotation'];c2w[:3,3]=c['position'];view=np.linalg.inv(c2w).astype(np.float32).T
        tx,ty=c['width']/(2*c['fx']),c['height']/(2*c['fy'])
        files.extend(ROOT/m['path'] for m in old['models'] if m['scene']==scene)
        files.extend(V12/scene/'fresh/main/sh_only'/n for n in ['rgb.npy','aux.npy','radii.npy','maxblend.npy','contributors.npy'])
        for event in [3,4]:
            path=OLD/scene/f'event_{event}/input.npz';files.append(path);saved=np.load(path);parent=int(saved['parent_id'].item());xyz=saved['parent_points'].astype(float)@view[:3,:3].astype(float)+view[3,:3].astype(float)
            pts=[];box=None
            if np.all(xyz[:,2]>0):
                xy=np.column_stack([(xyz[:,0]/xyz[:,2]/tx+1)*W/2-.5,(xyz[:,1]/xyz[:,2]/ty+1)*H/2-.5])
                x0=max(0,math.floor(xy[:,0].min()));y0=max(0,math.floor(xy[:,1].min()));x1=min(W,math.ceil(xy[:,0].max())+1);y1=min(H,math.ceil(xy[:,1].max())+1)
                if x1>x0 and y1>y0:
                    box=[x0,y0,x1,y1]
                    for j in [1,2,3]:
                        for i in [1,2,3]:
                            if i==j==2:continue
                            p=[min(x1-1,x0+(x1-x0)*i//4),min(y1-1,y0+(y1-y0)*j//4)]
                            if p not in pts:pts.append(p)
            directed=len(pts);k=0
            while len(pts)<16:
                h=hashlib.sha256(f'v18fresh/{scene}/{event}/{k}'.encode()).digest();p=[int.from_bytes(h[:8],'little')%W,int.from_bytes(h[8:16],'little')%H];k+=1
                if p not in pts:pts.append(p)
            cases.append(dict(scene=scene,event=event,parent_id=parent,camera=c['img_name'],pixels=pts,directed_pixels=directed,projected_bbox=box,resolution=[W,H]))
    native=old['modules']['native'];obs=json.loads((ROOT/'experiments/tgs_window_scene_v15/trace/build_receipt.json').read_text())
    assert sha(Path(native['path']))==native['sha256'] and sha(Path(obs['path']))==obs['sha256']
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},native=native,observer=obs,cases=cases,
        full_forward_cap=6,trace_call_cap=256,training=False))
    print(json.dumps(dict(plan_frozen=True,cases=len(cases),pixels=sum(len(x['pixels']) for x in cases))),flush=True)

@torch.no_grad()
def run():
    plan=json.loads((OUT/'plan.json').read_text());assert not (OUT/'results.json').exists()
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    for key in ['native','observer']:assert sha(Path(plan[key]['path']))==plan[key]['sha256']
    native=module(plan['native']['path']);observer=module(plan['observer']['path']);helper=module(HELPER);forwards=traces=0;rows=[];baseline=[]
    def endpoint(res,state,cam,col,case,point,mode):
        nonlocal traces
        x,y=point;packet,ids,ep,meta=gather(res,state,cam,col,x,y);p=torch.from_numpy(packet.copy()).cuda()
        actual,summary=observer.replay(p,x,y,False);full,fullsummary=observer.replay(p,x,y,True);torch.cuda.synchronize();traces+=2
        summary=summary.cpu().numpy();assert exact(summary[:5],ep.astype(np.float32))
        name=f"{case['scene']}_e{case['event']}_{x}_{y}_{mode}.npz"
        np.savez_compressed(OUT/name,packed=packet,actual=actual.cpu().numpy(),summary=summary,full=full.cpu().numpy(),fullsummary=fullsummary.cpu().numpy(),native=ep,
            parent_color=col[case['parent_id'] if mode=='before' else len(state['points'])-1].cpu().numpy())
        return name
    for scene in ['flowers','room']:
        w,camera_rows=workload(scene);ci=next(i for i,x in enumerate(camera_rows) if x['cohort']=='fresh');cam=w.cameras[ci];initial=w.input_hashes()
        state=pack(w.points,w.sigma,w.opacity,w.sh);col=colors(helper,state,cam);cases=[c for c in plan['cases'] if c['scene']==scene]
        res=call(native,state,cam,col);torch.cuda.synchronize();forwards+=1;arrays=capture(res,cam['w'],cam['h']);prior=V12/scene/'fresh/main/sh_only'
        for k in ['aux','radii','maxblend','contributors']:assert exact(arrays[k],np.load(prior/(k+'.npy'))),(scene,k)
        rgb_error=float(np.max(np.abs(arrays['rgb'].astype(float)-np.load(prior/'rgb.npy').astype(float))));assert rgb_error<=2e-4
        baseline.append(dict(scene=scene,RGB_precomputed_vs_SH_maxabs=rgb_error,noncolor_literal=True,hashes={k:ahash(a) for k,a in arrays.items()}))
        old_names={}
        for case in cases:
            for point in case['pixels']:old_names[(case['event'],*point)]=endpoint(res,state,cam,col,case,point,'before')
        del res,arrays
        for case in cases:
            index=case['parent_id'];changed=split(state,index);cc=torch.cat([col[:index],col[index+1:],col[index:index+1].repeat(4,1)]).contiguous()
            res=call(native,changed,cam,cc);torch.cuda.synchronize();forwards+=1;arrays=capture(res,cam['w'],cam['h'])
            write_new(OUT/f"{scene}_e{case['event']}_native_hashes.json",{k:ahash(a) for k,a in arrays.items()})
            for point in case['pixels']:
                name=endpoint(res,changed,cam,cc,case,point,'after')
                rows.append(dict(scene=scene,camera=case['camera'],event=case['event'],parent_id=index,N=len(state['points']),pixel=point,before=old_names[(case['event'],*point)],after=name))
            del res,arrays,changed,cc
            print(json.dumps(dict(scene=scene,event=case['event'],full_forwards=forwards,trace_calls=traces)),flush=True)
        assert initial==w.input_hashes()
        del w,state,col;gc.collect();torch.cuda.empty_cache()
    assert forwards==6 and traces==256 and len(rows)==64
    write_new(OUT/'results.json',dict(rows=rows,baselines=baseline,full_forwards=forwards,trace_calls=traces,
        all_native_endpoints_literal=True,training=False,plan_sha256=sha(OUT/'plan.json')))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
