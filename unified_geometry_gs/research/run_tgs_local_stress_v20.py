"""Frozen-code coverage/appearance stress for the v19 single-parent query."""
import argparse,gc,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,ahash,write_new,module,pack,split,colors,exact
from tgs_sh_isolation_common_v12 import capture,workload
from trace_tgs_window_scene_v15 import call
from run_tgs_local_split_v19 import cache,cache_gate,same_bits,load_local
OUT=ROOT/'experiments/tgs_local_stress_v20';V19=ROOT/'experiments/tgs_local_split_v19'

def prepare():
    OUT.mkdir(exist_ok=True);old=json.loads((V19/'plan.json').read_text());assert json.loads((V19/'results.json').read_text())['all_gates_passed']
    files=[Path(__file__),ROOT/'TGS_LOCAL_STRESS_PROTOCOL_V20.md',ROOT/'run_tgs_local_split_v19.py',ROOT/'tgs_local_split_v19.cu',
        ROOT/'tgs_split_color_common_v14.py',ROOT/'tgs_sh_isolation_common_v12.py',ROOT/'trace_tgs_window_scene_v15.py',
        ROOT/'tgs_cross_scene_common_v10.py',ROOT/'run_tgs_real_events.py',HELPER,V19/'plan.json',V19/'build_receipt.json',V19/'evidence_manifest.json',
        ROOT/'experiments/tgs_sh_isolation_v12/input_plan.json',ROOT/'experiments/tgs_sh_isolation_v12/identity_audit.json',
        ROOT/'experiments/tgs_coupled_v18/fresh/results.json']
    files.extend(ROOT/p for p in old['inputs'] if p.endswith('point_cloud_state_dict.pt'))
    local=json.loads((V19/'build_receipt.json').read_text());assert sha(Path(local['path']))==local['sha256']
    native=old['native'];assert sha(Path(native['path']))==native['sha256']
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},native=native,local=local,
        targets=[1,.5,.1,.01],scenes=['flowers','room'],appearance=['hold','moved'],full_forward_cap=26,training=False))
    print('STRESS_PLAN_FROZEN',flush=True)

def select(bg,state,scene,cam):
    visible=(bg[6]>0)&(state['opacity'].flatten()>=.01);ids=torch.nonzero(visible).flatten();lo=bg[7][ids].cpu().numpy();hi=bg[8][ids].cpu().numpy()
    ids=ids.cpu().numpy();op=state['opacity'].flatten()[torch.from_numpy(ids).cuda()].cpu().numpy();area=np.prod(hi.astype(np.int64)-lo.astype(np.int64),axis=1)
    assert len(ids)>=4 and np.all(area>0)
    assert np.all(lo>=0) and np.all(hi[:,0]<=(cam['w']+15)//16) and np.all(hi[:,1]<=(cam['h']+15)//16)
    chosen=[];amax=int(area.max())
    for fraction in [1,.5,.1,.01]:
        candidates=np.flatnonzero(~np.isin(ids,np.array([r['parent_id'] for r in chosen],dtype=ids.dtype)))
        distances=np.abs(area[candidates].astype(float)-amax*fraction);order=np.lexsort((ids[candidates],distances));ix=int(candidates[order[0]])
        chosen.append(dict(parent_id=int(ids[ix]),target_fraction_of_max=fraction,parent_tiles=int(area[ix]),rect_min=lo[ix].tolist(),rect_max=hi[ix].tolist(),opacity=float(op[ix])))
    folder=OUT/scene;folder.mkdir();np.savez_compressed(folder/'eligible.npz',ids=ids,area=area,lo=lo,hi=hi,opacity=op)
    write_new(folder/'selection.json',dict(scene=scene,camera=cam.get('name'),resolution=[cam['w'],cam['h']],max_parent_tiles=amax,eligible=len(ids),
        eligible_sha256=sha(folder/'eligible.npz'),parents=chosen,before_all_split_outputs=True))
    return chosen

def save_patch(folder,base,local,mask):
    H,W=mask.shape;yy,xx=np.nonzero(mask)
    box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else [0,0,0,0];x0,y0,x1,y1=box
    np.savez_compressed(folder/'query_patch.npz',bbox=np.array(box),shape=np.array([H,W]),rgb=local[0][:,y0:y1,x0:x1],alpha=local[1][y0:y1,x0:x1],paths=local[2][:,y0:y1,x0:x1])
    for i,k in enumerate(['rgb','alpha','paths']):
        a=base[k].copy()
        if i==1:a[y0:y1,x0:x1]=local[i][y0:y1,x0:x1]
        else:a[:,y0:y1,x0:x1]=local[i][:,y0:y1,x0:x1]
        assert exact(a,local[i])
    return box

@torch.no_grad()
def run():
    plan=json.loads((OUT/'plan.json').read_text());assert not (OUT/'results.json').exists()
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    for name in ['native','local']:assert sha(Path(plan[name]['path']))==plan[name]['sha256']
    native=module(plan['native']['path']);m=load_local();helper=module(HELPER);prior=json.loads((ROOT/'experiments/tgs_coupled_v18/fresh/results.json').read_text());rows=[];forwards=0
    for scene in plan['scenes']:
        w,camera_rows=workload(scene);ci=next(i for i,r in enumerate(camera_rows) if r['cohort']=='fresh');cam=w.cameras[ci];cam['name']=camera_rows[ci]['camera']['img_name'];initial=w.input_hashes()
        state=pack(w.points,w.sigma,w.opacity,w.sh);col=colors(helper,state,cam);base=call(native,state,cam,col);torch.cuda.synchronize();forwards+=1
        arrays=capture(base,cam['w'],cam['h']);expected=next(x for x in prior['baselines'] if x['scene']==scene)['hashes']
        for k,a in arrays.items():assert ahash(a)==expected[k],(scene,k)
        bg,br,bl,bpaths=cache(base,state,cam);alpha=base[2][1].contiguous();W,H=cam['w'],cam['h'];tx,ty=(W+15)//16,(H+15)//16;tiles=tx*ty
        parents=select(bg,state,scene,cam);folder=OUT/scene;np.savez_compressed(folder/'baseline.npz',rgb=arrays['rgb'],alpha=arrays['aux'][1],paths=arrays['contributors'].view(np.int32))
        basecpu=dict(rgb=arrays['rgb'],alpha=arrays['aux'][1],paths=arrays['contributors'].view(np.int32));del arrays
        yy,xx=torch.meshgrid(torch.arange(ty,device='cuda'),torch.arange(tx,device='cuda'),indexing='ij');tilepixels=(W-xx*16).clamp(max=16)*(H-yy*16).clamp(max=16)
        for pi,parent in enumerate(parents):
            index=parent['parent_id'];changed=split(state,index);child=pack(changed['points'][-4:],changed['sigma'][-4:],changed['opacity'][-4:],changed['sh'][-4:]);cc_hold=col[index:index+1].repeat(4,1).contiguous();cc_moved=colors(helper,child,cam)
            pf=folder/f'parent_{pi}';pf.mkdir();np.savez_compressed(pf/'input.npz',parent_id=np.array(index),parent_points=state['points'][index].cpu().numpy(),child_points=child['points'].cpu().numpy(),sigma=child['sigma'].cpu().numpy(),opacity=child['opacity'].cpu().numpy(),hold_colors=cc_hold.cpu().numpy(),moved_colors=cc_moved.cpu().numpy())
            small=call(native,child,cam,cc_hold);torch.cuda.synchronize();forwards+=1;cg,cr,cl,_=cache(small,child,cam);hold_arrays=None
            for appearance,cc in [('hold',cc_hold),('moved',cc_moved)]:
                q=m.query(bg,br,bl,col,cg,cr,cl,cc,index,base[1],alpha,bpaths,W,H);torch.cuda.synchronize()
                fc=torch.cat([col[:index],col[index+1:],cc]).contiguous();full=call(native,changed,cam,fc);torch.cuda.synchronize();forwards+=1;fa=capture(full,W,H);fg,fr,fl,fp=cache(full,changed,cam)
                cache_gate(bg,cg,fg,index);stats=m.audit(bg,br,bl,cg,cr,cl,fr,fl,index,len(state['points']),tiles);torch.cuda.synchronize();sv=stats.cpu().tolist();assert not any(sv[:4]),sv
                if appearance=='hold':hold_arrays={k:a for k,a in fa.items() if k!='rgb'}
                else:
                    for k,a in hold_arrays.items():assert exact(a,fa[k]),(scene,pi,k)
                checks=dict(rgb=same_bits(q[0],full[1]),alpha=same_bits(q[1],full[2][1]),paths=same_bits(q[2],fp))
                mask=q[3].reshape(ty,tx).repeat_interleave(16,0).repeat_interleave(16,1)[:H,:W].bool()
                checks['outside_rgb']=same_bits(q[0][:,~mask],base[1][:,~mask]);checks['outside_alpha']=same_bits(q[1][~mask],alpha[~mask]);checks['outside_paths']=same_bits(q[2][:,~mask],bpaths[:,~mask])
                query=[a.cpu().numpy() for a in q[:3]];paf=pf/appearance;paf.mkdir();maskcpu=mask.cpu().numpy();box=save_patch(paf,basecpu,query,maskcpu)
                lengths=(fr.reshape(tiles,2)[:,1]-fr.reshape(tiles,2)[:,0]).long().reshape(ty,tx);affected=q[3].reshape(ty,tx).bool();work=int((lengths[affected]*tilepixels[affected]).sum().item());allwork=int((lengths*tilepixels).sum().item())
                hashes={k:ahash(a) for k,a in zip(['rgb','alpha','paths'],query)}
                native_hashes=dict(rgb=ahash(fa['rgb']),alpha=ahash(fa['aux'][1]),paths=ahash(fa['contributors'].view(np.int32)))
                receipt=dict(scene=scene,camera=cam['name'],parent_index=pi,parent_id=index,appearance=appearance,checks=checks,query_hashes=hashes,native_hashes=native_hashes,
                    all_defined_cache_fields_literal=True,all_tile_lists_literal=True,tile_audit=sv,affected_tiles=int(q[3].sum().item()),affected_pixels=int(mask.sum().item()),pixels=W*H,
                    candidate_pixel_visits_upper=work,all_candidate_pixel_visits_upper=allwork,bbox=box,child_color_delta_max=float((cc_moved-cc_hold).abs().max().item()),native_forwards=forwards)
                write_new(paf/'receipt.json',receipt)
                if not all(checks.values()):np.savez_compressed(paf/'native_mismatch.npz',rgb=fa['rgb'],alpha=fa['aux'][1],paths=fa['contributors'].view(np.int32));raise AssertionError(receipt)
                rows.append(receipt);print(json.dumps(dict(scene=scene,parent_index=pi,appearance=appearance,literal=True,coverage=receipt['affected_pixels']/(W*H),native_forwards=forwards)),flush=True)
                del q,fc,full,fa,fg,fr,fl,fp,stats,mask,query
            del changed,child,cc_hold,cc_moved,small,cg,cr,cl,hold_arrays
        assert initial==w.input_hashes()
        del w,state,col,base,bg,br,bl,bpaths,alpha,basecpu;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==16 and forwards==26
    write_new(OUT/'results.json',dict(rows=rows,full_forwards=forwards,all_gates_passed=True,training=False,timing_not_yet_performed=True));print('STRESS_ALL16_LITERAL',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
