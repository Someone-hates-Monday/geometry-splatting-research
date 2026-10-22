"""Exactly20 forward-only references for frozen parents in a second known camera."""
import argparse,gc,json,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tgs_split_color_common_v14 import ROOT,HELPER,sha,ahash,write_new,module,pack,split,colors,CrossWorkload,exact
from tgs_aux_ablation_common_v21 import control
from tgs_batch_score_common_v22 import backend,bank,schedule,cpu_sse
from run_tgs_batch_score_v22 import score
from run_tgs_local_split_v19 import cache,cache_gate,same_bits
from trace_tgs_window_scene_v15 import call
from tgs_image_metrics_v11 import png8
OUT=ROOT/'experiments/tgs_view_transfer_v23';V20=ROOT/'experiments/tgs_local_stress_v20';V22=ROOT/'experiments/tgs_batch_score_v22'

def prepare():
    OUT.mkdir();prior=json.loads((V22/'results.json').read_text());assert prior['all_gates_passed']
    files=[Path(__file__),ROOT/'analyze_tgs_view_transfer_v23.py',ROOT/'TGS_VIEW_TRANSFER_PROTOCOL_V23.md',
        ROOT/'tgs_aux_ablation_common_v21.py',ROOT/'tgs_batch_score_common_v22.py',ROOT/'run_tgs_batch_score_v22.py',
        ROOT/'run_tgs_local_split_v19.py',ROOT/'tgs_split_color_common_v14.py',ROOT/'trace_tgs_window_scene_v15.py',
        ROOT/'tgs_image_metrics_v11.py',ROOT/'tgs_cross_scene_common_v10.py',ROOT/'run_tgs_real_events.py',HELPER,
        V22/'results.json',V22/'evidence_manifest.json',V22/'build_receipt.json',V20/'plan.json',V20/'evidence_manifest.json',
        ROOT/'experiments/tgs_aux_ablation_v21/build_receipt.json',ROOT/'experiments/tgs_gt_quality_v11/identity_audit.json',
        ROOT/'experiments/tgs_gt_quality_v11/evidence_manifest.json',ROOT/'experiments/tgs_sh_isolation_v12/identity_audit.json',
        ROOT/'sources/tgs_subdivision_static_v1/manifest.json']
    source=ROOT/'sources/tgs_subdivision_static_v1/triangle-splatting'
    files.extend(source/n for n in ['train.py','utils/loss_utils.py','arguments/__init__.py'])
    cases=json.loads((V22/'plan.json').read_text())['cases']
    for scene in ['flowers','room']:
        model=ROOT/f'sources/tgs_cross_scene_v10/{scene}'
        files.extend(model/n for n in ['point_cloud/iteration_7000/point_cloud_state_dict.pt','point_cloud/iteration_7000/hyperparameters.pt','cfg_args','cameras.json'])
        files.extend([V20/scene/'baseline.npz',ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy',ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy',ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy'])
        for prefix in [ROOT/f'sources/tgs_gt_identity_v11/{scene}',ROOT/f'sources/tgs_sh_isolation_v12/{scene}/fresh']:
            files.extend(prefix/n for n in ['original.JPG','published_gt.png'])
        for i in range(4):
            files.append(V20/scene/f'parent_{i}/input.npz')
            files.extend(V20/scene/f'parent_{i}'/app/'query_patch.npz' for app in ['hold','moved'])
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},cases=cases,new_view='old_0',full_forward_cap=20,training=False,accepted_mutations=0))
    print('VIEW_TRANSFER_PLAN_FROZEN',flush=True)

def frozen():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p

def gt_gate(scene):
    gt=np.load(ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy')
    for name in ['original.JPG','published_gt.png']:
        assert np.array_equal(gt,np.asarray(Image.open(ROOT/f'sources/tgs_gt_identity_v11/{scene}'/name).convert('RGB')))
    return gt

@torch.no_grad()
def run():
    plan=frozen();m=backend();ctrl=control();helper=module(HELPER);rows=[];forwards=0
    identity=json.loads((ROOT/'experiments/tgs_gt_quality_v11/identity_audit.json').read_text());assert identity['all_gates']
    for scene in ['flowers','room']:
        folder=OUT/scene;folder.mkdir();w=CrossWorkload(scene);initial=w.input_hashes();cam=w.cameras[0];W,H=cam['w'],cam['h']
        camera=w.camera_metadata[0]['img_name'];assert camera==next(x['camera'] for x in identity['rows'] if x['scene']==scene)
        state=pack(w.points,w.sigma,w.opacity,w.sh);col=colors(helper,state,cam);base=call(ctrl,state,cam,col);torch.cuda.synchronize();forwards+=1
        bg,br,bl,paths=cache(base,state,cam);alpha=base[2][1].contiguous();original=base[1];rgb=original.cpu().numpy()
        assert exact(rgb,np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy'))
        gt=gt_gate(scene);assert gt.dtype==np.uint8 and gt.shape==(H,W,3)
        gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda();base8=m.quantize(original);torch.cuda.synchronize()
        assert np.array_equal(base8.cpu().numpy().transpose(1,2,0),png8(rgb));base_sse=cpu_sse(png8(rgb),gt)
        cases=[x for x in plan['cases'] if x['scene']==scene];bs,cc,parents,small,cc_list=bank(state,col,cam,cases,helper)
        np.savez_compressed(folder/'bank_inputs.npz',parents=parents.cpu().numpy(),points=bs['points'].cpu().numpy(),sigma=bs['sigma'].cpu().numpy(),opacity=bs['opacity'].cpu().numpy(),sh=bs['sh'].cpu().numpy(),colors=cc.cpu().numpy())
        batch=call(ctrl,bs,cam,cc);torch.cuda.synchronize();forwards+=1;cg,_,_,_=cache(batch,bs,cam);boxes,offsets=schedule(m,bg,cg,parents,W,H)
        allout=score(m,bg,br,bl,col,cg,cc,parents,boxes,offsets,base8,gt8,original,alpha,paths,W,H);torch.cuda.synchronize()
        assert not any(allout[1].cpu().tolist()) and all(x.numel()==0 for x in allout[2:])
        scene_rows=[]
        for e,case in enumerate(cases):
            parent=case['parent_id'];changed=split(state,parent);assert same_bits(changed['points'][-4:],small[e]['points'])
            fc=torch.cat([col[:parent],col[parent+1:],cc_list[e]]).contiguous();full=call(ctrl,changed,cam,fc);torch.cuda.synchronize();forwards+=1
            fg,fr,fl,fp=cache(full,changed,cam);cache_gate(bg,[g[4*e:4*e+4] for g in cg],fg,parent)
            stats=m.audit(bg,br,bl,cg,parents,e,fr,fl,len(state['points']),W,H)
            debug=score(m,bg,br,bl,col,cg,cc,parents,boxes,offsets,base8,gt8,original,alpha,paths,W,H,e);native_delta=m.delta(full[1],base8,gt8);torch.cuda.synchronize()
            assert not any(stats[:4].cpu().tolist()) and not any(debug[1].cpu().tolist())
            checks=dict(rgb=same_bits(debug[2],full[1]),alpha=same_bits(debug[3],full[2][1]),paths=same_bits(debug[4],fp));assert all(checks.values())
            fullrgb=full[1].cpu().numpy();delta=cpu_sse(png8(fullrgb),gt)-base_sse
            assert delta==int(allout[0][e].item())==int(debug[0][e].item())==int(native_delta.item())
            box=boxes[e].cpu().tolist();x0,y0,x1,y1=box[0]*16,box[1]*16,min(W,box[2]*16),min(H,box[3]*16)
            patch=fullrgb[:,y0:y1,x0:x1].copy();rebuilt=rgb.copy();rebuilt[:,y0:y1,x0:x1]=patch;assert exact(rebuilt,fullrgb)
            np.savez_compressed(folder/f'proposal_{e}_rgb_patch.npz',bbox=np.array([x0,y0,x1,y1]),rgb=patch)
            row=dict(scene=scene,view='old_0',camera=camera,event=e,parent_id=parent,parent_index=case['parent_index'],appearance=case['appearance'],
                checks=checks,visible_cache_fields_literal=True,all_tile_lists_literal=True,delta_SSE=delta,baseline_SSE=base_sse,
                full_rgb_sha256=ahash(fullrgb),base_rgb_sha256=ahash(rgb),changed_RGB_pixels=int(np.any(fullrgb!=rgb,axis=0).sum()),bbox=[x0,y0,x1,y1],
                batch_score=int(allout[0][e].item()),full_gpu_score=int(native_delta.item()),native_forwards=forwards)
            write_new(folder/f'proposal_{e}.json',row);rows.append(row);scene_rows.append(row)
            print(json.dumps(dict(scene=scene,event=e,exact=True,delta_SSE=delta,native_forwards=forwards)),flush=True)
            del changed,fc,full,fg,fr,fl,fp,debug,stats,native_delta,fullrgb,patch,rebuilt
        assert initial==w.input_hashes()
        write_new(folder/'receipt.json',dict(rows=scene_rows,model_inputs_unchanged=True,camera_identity=True,GT_identity=True))
        del w,state,col,base,bg,br,bl,paths,alpha,original,rgb,gt8,base8,bs,cc,parents,small,cc_list,batch,cg,boxes,offsets,allout;gc.collect();torch.cuda.empty_cache()
    assert forwards==20 and len(rows)==16
    write_new(OUT/'results.json',dict(all_passed=True,rows=rows,native_forwards=forwards,training=False,accepted_mutations=0,source_sha256=sha(Path(__file__))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
