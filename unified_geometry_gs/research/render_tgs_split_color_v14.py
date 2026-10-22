"""76-forward cap, two frozen real models, no_grad and no optimizer; color intervention only."""
import gc,hashlib,heapq,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,OUT,HELPER,CrossWorkload,pack,split,sha,ahash,write_new,module,frozen_modules,render,colors,exact,save_case

def save_arrays(folder,arrays):
    folder.mkdir()
    for key,value in arrays.items():np.save(folder/(key+'.npy'),value,allow_pickle=False)

def hashes(arrays):return {k:dict(sha256=ahash(v),dtype=str(v.dtype),shape=list(v.shape)) for k,v in arrays.items()}

@torch.no_grad()
def main():
    assert not (OUT/'results.json').exists()
    plan=json.loads((OUT/'plan.json').read_text())
    for n,h in plan['scripts'].items():assert sha(ROOT/n)==h
    assert sha(HELPER)==plan['helper_sha256']
    assert sha(ROOT/'TGS_SPLIT_COLOR_PROTOCOL_V14.md')==plan['protocol_sha256']
    mods=frozen_modules(plan);helper=module(str(HELPER));rows=[];baseline_rows=[];forwards=0
    for scene in ['flowers','room']:
        model=next(m for m in plan['models'] if m['scene']==scene);assert sha(ROOT/model['path'])==model['sha256']
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh)
        sc=next(s for s in plan['scenes'] if s['scene']==scene);assert w.camera_metadata==sc['cameras']
        base=OUT/scene;base.mkdir();baselines=[];baseline_colors=[]
        for ci,cam in enumerate(w.cameras):
            col=colors(helper,state,cam);before,tiles,_=render(mods['native'],state,cam,col);forwards+=1
            repeat,repeat_tiles,_=render(mods['native'],state,cam,col);forwards+=1
            checks={k:exact(before[k],repeat[k]) for k in before};assert all(checks.values()) and tiles==repeat_tiles
            native,native_tiles,_=render(mods['sh_only'],state,cam);forwards+=1
            noncolor={k:exact(before[k],native[k]) for k in before if k!='rgb'}
            error=float(np.abs(before['rgb'].astype(float)-native['rgb'].astype(float)).max())
            assert error<=2e-4 and all(noncolor.values()) and tiles==native_tiles
            folder=base/f'baseline_{ci}';save_arrays(folder,before)
            np.save(folder/'colors.npy',col.cpu().numpy(),allow_pickle=False)
            np.savez(folder/'camera.npz',view=cam['view'].cpu().numpy(),projection=cam['projection'].cpu().numpy(),center=cam['center'].cpu().numpy())
            row=dict(scene=scene,camera=ci,camera_name=w.camera_metadata[ci]['img_name'],tiles=tiles,
                repeat_literal=checks,native_SH_only_noncolor_literal=noncolor,native_SH_only_RGB_maxabs=error,
                arrays=hashes(before),color_sha256=ahash(col.cpu().numpy()))
            baseline_rows.append(row);write_new(folder/'receipt.json',row)
            baselines.append(before);baseline_colors.append(col)
            print(json.dumps(dict(stage='baseline',scene=scene,camera=ci,SH_check=error,forwards=forwards)),flush=True)
            del repeat,native
        max_radius=np.maximum.reduce([b['radii'] for b in baselines]);op=state['opacity'].cpu().numpy().reshape(-1)
        eligible=np.flatnonzero((max_radius>24)&(op>=.01))
        key=lambda i:hashlib.sha256(f'tgs_split_color_v14/{scene}/{int(i)}'.encode()).hexdigest()
        ids=heapq.nsmallest(8,eligible,key=key);assert len(ids)==8
        np.save(base/'eligible_ids.npy',eligible,allow_pickle=False)
        selection=dict(scene=scene,chosen_ids=[int(i) for i in ids],eligible_count=len(eligible),
            eligible_sha256=ahash(eligible),rule='min SHA256(tgs_split_color_v14/scene/integerID)',
            before_any_split_forward=True,model_inputs=initial)
        write_new(base/'selection.json',selection)
        # All event tensors and source fields saved before this scene's first split rendering.
        for ei,index in enumerate(ids):
            changed=split(state,int(index));folder=base/f'event_{ei}';folder.mkdir()
            np.savez(folder/'input.npz',parent_id=np.array(index),parent_points=state['points'][index].cpu().numpy(),
                child_points=changed['points'][-4:].cpu().numpy(),sigma=state['sigma'][index].cpu().numpy(),
                opacity=state['opacity'][index].cpu().numpy(),SH=state['sh'][index].cpu().numpy())
            del changed
        for ei,index in enumerate(ids):
            changed=split(state,int(index))
            for ci,cam in enumerate(w.cameras):
                before=baselines[ci];col=baseline_colors[ci];folder=base/f'event_{ei}'/f'camera_{ci}';folder.mkdir()
                holdcol=torch.cat([col[:index],col[index+1:],col[index:index+1].repeat(4,1)]).contiguous()
                movedcol=holdcol.clone();child_d=changed['points'][-4:].mean(1)-cam['center'][None]
                child_d/=child_d.norm(dim=1,keepdim=True)
                movedcol[-4:]=(helper.eval_sh(3,changed['sh'][-4:].transpose(1,2),child_d)+.5).clamp_min_(0)
                assert torch.equal(holdcol[:-4],movedcol[:-4])
                hold,htiles,hinradius=render(mods['native'],changed,cam,holdcol,child_tail=True);forwards+=1
                moved,mtiles,minradius=render(mods['native'],changed,cam,movedcol,child_tail=True);forwards+=1
                checks={k:exact(hold[k],moved[k]) for k in hold if k!='rgb'}
                checks['tile_count']=htiles==mtiles;assert all(checks.values())
                assert hinradius==minradius
                box,outcomes=save_case(folder,before,hold,moved)
                np.savez_compressed(folder/'noncolor.npz',**{v+'_'+k:a[k] for v,a in [('hold',hold),('moved',moved)] for k in ['radii','maxblend','contributors']})
                np.savez(folder/'colors.npz',parent=col[index].cpu().numpy(),hold=holdcol[-4:].cpu().numpy(),moved=movedcol[-4:].cpu().numpy(),child_directions=child_d.cpu().numpy())
                row=dict(scene=scene,event=ei,parent_id=int(index),camera=ci,camera_name=w.camera_metadata[ci]['img_name'],
                    split_noncolor_literal=checks,hold_tiles=htiles,moved_tiles=mtiles,
                    hold_hashes=hashes(hold),moved_hashes=hashes(moved),bbox=box,metrics=outcomes,
                    parent_radii=int(before['radii'][index]),child_radii=hold['radii'][-4:].tolist(),
                    visible_child_inradius=hinradius,child_color_maxabs=float(torch.abs(movedcol[-4:]-holdcol[-4:]).max().item()))
                write_new(folder/'receipt.json',row);rows.append(row)
                (OUT/'progress.json').write_text(json.dumps(dict(forwards=forwards,cases=rows),indent=2,allow_nan=False)+'\n')
                print(json.dumps(dict(stage='split',scene=scene,event=ei,camera=ci,forwards=forwards,
                    appearance_peak=outcomes['appearance']['peak'],noncolor_peak=outcomes['noncolor']['peak'])),flush=True)
                del hold,moved,holdcol,movedcol
            del changed
        assert w.input_hashes()==initial and sha(ROOT/model['path'])==model['sha256']
        del w,state,baselines,baseline_colors,col;gc.collect();torch.cuda.empty_cache()
    assert len(rows)==32 and forwards==76
    write_new(OUT/'results.json',dict(cases=rows,baselines=baseline_rows,forwards=forwards,
        all_isolation_gates=True,model_inputs_unchanged=True,plan_sha256=sha(OUT/'plan.json'),
        script_sha256=sha(Path(__file__)),training=False,benchmark=False,real_training_events=False,
        actual_captured_cameras=True,real_image_quality_gain_established=False))
    print(json.dumps(dict(complete=True,cases=len(rows),forwards=forwards,training=False)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());raise
