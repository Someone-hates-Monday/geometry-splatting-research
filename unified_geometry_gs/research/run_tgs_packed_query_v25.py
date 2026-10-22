"""Eight new native forwards integrate packed replay with immutable FP64 loss."""
import argparse,gc,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tgs_split_color_common_v14 import ROOT,HELPER,sha,ahash,write_new,module,pack,colors,CrossWorkload,exact
from tgs_aux_ablation_common_v21 import control,view_list
from tgs_batch_score_common_v22 import backend as planner_backend,bank
from tgs_local_image_loss_common_v24 import backend as loss_backend,evaluate
from tgs_packed_query_common_v25 import OUT,backend,dynamic_layout,emit,query
from run_tgs_local_split_v19 import cache,fake
from run_tgs_batch_score_v22 import composite
from trace_tgs_window_scene_v15 import call
V20=ROOT/'experiments/tgs_local_stress_v20';V22=ROOT/'experiments/tgs_batch_score_v22';V23=ROOT/'experiments/tgs_view_transfer_v23';V24=ROOT/'experiments/tgs_local_image_loss_v24'

def prepare():
    OUT.mkdir();assert json.loads((V24/'results.json').read_text())['all_passed']
    files=[Path(__file__),ROOT/'tgs_packed_patch_v25.cu',ROOT/'tgs_packed_query_common_v25.py',ROOT/'TGS_PACKED_QUERY_PROTOCOL_V25.md',
        ROOT/'tgs_batch_score_v22.cu',ROOT/'tgs_batch_score_common_v22.py',ROOT/'tgs_local_image_loss_common_v24.py',ROOT/'run_tgs_batch_score_v22.py',
        ROOT/'run_tgs_local_split_v19.py',ROOT/'tgs_split_color_common_v14.py',ROOT/'tgs_aux_ablation_common_v21.py',ROOT/'trace_tgs_window_scene_v15.py',
        ROOT/'tgs_cross_scene_common_v10.py',ROOT/'run_tgs_real_events.py',ROOT/'tgs_image_metrics_v11.py',HELPER,
        V22/'plan.json',V22/'build_receipt.json',V23/'analysis.json',V23/'evidence_manifest.json',V24/'results.json',V24/'build_receipt.json',V24/'evidence_manifest.json',
        ROOT/'experiments/tgs_aux_ablation_v21/build_receipt.json']
    for old in [V23/'plan.json',V24/'plan.json']:
        p=json.loads(old.read_text());files.extend(ROOT/n for n in p['inputs'])
    for scene in ['flowers','room']:files.extend([V22/scene/'bank_inputs.npz',V23/scene/'bank_inputs.npz'])
    files=sorted(set(files));write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},cases=json.loads((V22/'plan.json').read_text())['cases'],
        views=['fresh','old_0'],new_native_forward_cap=8,new_complete_candidate_reference_forwards=0,training=False,accepted_mutations=0))
    print('PACKED_QUERY_PLAN_FROZEN',flush=True)

def frozen():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    return p

def build():
    frozen();folder=OUT/'build';folder.mkdir();os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'];m=load(name='tgs_packed_patch_v25',sources=[str(ROOT/'tgs_packed_patch_v25.cu')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(folder),verbose=True)
    write_new(OUT/'build_receipt.json',dict(path=m.__file__,sha256=sha(Path(m.__file__)),source_sha256=sha(ROOT/'tgs_packed_patch_v25.cu'),
        flags=['-O3','--use_fast_math','sm_120'],torch=torch.__version__,cuda=torch.version.cuda,training=False));print('PACKED_QUERY_BUILD_DONE',flush=True)

def fixture_check(m,planner,name,bg,br,bl,bc,cg,cc,parents,base,targets):
    H,W=base.shape[1:];boxes,offsets,d,rois=dynamic_layout(planner,bg,cg,parents,W,H);out=emit(m,bg,br,bl,bc,cg,cc,parents,boxes,offsets,base,d);torch.cuda.synchronize();assert not any(out[1].cpu().tolist())
    raw=out[0].cpu().numpy();rows=[]
    for e,(r,target) in enumerate(zip(rois,targets)):
        x0,y0,x1,y1=r;start,end=d['host']['prefix'][0][e:e+2];p=raw[int(start):int(end)].reshape(3,y1-y0,x1-x0)
        assert exact(p,target[:,y0:y1,x0:x1]),(name,e)
        rows.append(dict(event=e,ROI=r,literal=True,patch_hash=ahash(p)))
    return dict(name=name,rows=rows)

@torch.no_grad()
def fixtures():
    frozen();m=backend();planner=planner_backend();rows=[];ints=lambda a:torch.tensor(a,device='cuda',dtype=torch.int32)
    for name in ['stable_independent_deleted_empty','all257']:
        N,K=(4,4) if name.startswith('stable') else (258,1);bg=fake(N,[1.,2.,2.,3.] if N==4 else None,opacity=.5 if N==4 else 1/128)
        if N==4:bg[6][3]=0
        cg=fake(4*K,[2.]*(4*K));parents=ints([1,1,0,3] if K==4 else [0])
        if K==4:cg[6][8:]=0
        else:cg[6].zero_()
        oldids=[0,1,2] if N==4 else list(range(N));br=ints([0,len(oldids)]);bl=ints(oldids);bc=torch.ones((N,3),device='cuda');cc=torch.ones((4*K,3),device='cuda')
        if K==4:bc[:3]=torch.eye(3,device='cuda');cc[:4]=torch.tensor([0.,1.,0.],device='cuda');cc[4:8]=torch.tensor([[.25,.5,.75],[.5,.25,.125],[.125,.25,.5],[.75,.5,.25]],device='cuda')
        C,_,_=composite([(float(bg[4][i,3].item()),bc[i].cpu().numpy()) for i in oldids]);base=torch.from_numpy(C.copy()).cuda().reshape(3,1,1);targets=[]
        for e in range(K):
            parent=int(parents[e].item());items=[(float(bg[0][i].item()),0,i) for i in oldids if i!=parent];items.extend((2.,1,j) for j in range(4) if cg[6][4*e+j].item()>0);items.sort()
            ls=[(float((bg if kind==0 else cg)[4][i if kind==0 else 4*e+i,3].item()),(bc[i] if kind==0 else cc[4*e+i]).cpu().numpy()) for _,kind,i in items]
            t,_,_=composite(ls);targets.append(t.reshape(3,1,1))
        rows.append(fixture_check(m,planner,name,bg,br,bl,bc,cg,cc,parents,base,targets))
    bg=fake(2,[1.,3.]);cg=fake(4,[2.]*4,visible=False);cg[6][0]=1;bg[7][:]=ints([[0,0],[1,0]]);bg[8][:]=ints([[1,1],[2,1]]);cg[7][0]=ints([2,0]);cg[8][0]=ints([3,1])
    bc=torch.tensor([[1.,0.,0.],[0.,1.,0.]],device='cuda');cc=torch.tensor([[0.,0.,1.]]*4,device='cuda');base=torch.zeros((3,16,48),device='cuda');base[0,:,:16]=.5;base[1,:,16:32]=.5
    target=np.zeros((3,16,48),dtype=np.float32);target[1,:,16:32]=.5;target[2,:,32:]=.5
    rows.append(fixture_check(m,planner,'unaffected_middle_tile',bg,ints([0,1,1,2,2,2]),ints([0,1]),bc,cg,cc,ints([0]),base,[target]))
    write_new(OUT/'fixtures.json',dict(all_passed=True,software_only=True,rows=rows));print('PACKED_FIXTURES_PASSED',flush=True)

@torch.no_grad()
def validate():
    p=frozen();assert json.loads((OUT/'fixtures.json').read_text())['all_passed'];m=backend();planner=planner_backend();losser=loss_backend();ctrl=control();helper=module(HELPER)
    prior=json.loads((V23/'analysis.json').read_text());scores=json.loads((V24/'results.json').read_text());rows=[];forwards=0
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh)
        for view in p['views']:
            folder=OUT/f'{scene}_{view}';folder.mkdir();cam=next(v['cam'] for v in view_list(w,scene) if v['label']==view);W,H=cam['w'],cam['h'];col=colors(helper,state,cam)
            base=call(ctrl,state,cam,col);torch.cuda.synchronize();forwards+=1;bg,br,bl,_=cache(base,state,cam);rgb=base[1].cpu().numpy()
            expected_base=np.load(V20/scene/'baseline.npz')['rgb'] if view=='fresh' else np.load(ROOT/f'experiments/tgs_split_color_v14/{scene}/baseline_0/rgb.npy');assert exact(rgb,expected_base)
            gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy' if view=='fresh' else ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy');gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda()
            cases=[r for r in p['cases'] if r['scene']==scene];bs,cc,parents,small,cc_list=bank(state,col,cam,cases,helper)
            old=np.load((V22 if view=='fresh' else V23)/scene/'bank_inputs.npz')
            for key,value in [('points',bs['points']),('sigma',bs['sigma']),('opacity',bs['opacity']),('colors',cc),('parents',parents)]:assert exact(value.cpu().numpy(),old[key])
            child=call(ctrl,bs,cam,cc);torch.cuda.synchronize();forwards+=1;cg,_,_,_=cache(child,bs,cam)
            boxes,offsets,d,rois=dynamic_layout(planner,bg,cg,parents,W,H);patch=emit(m,bg,br,bl,col,cg,cc,parents,boxes,offsets,base[1],d)
            loss=evaluate(losser,base[1],gt8,patch[0],d,False);integrated=query(m,losser,planner,bg,br,bl,col,cg,cc,parents,base[1],gt8);torch.cuda.synchronize()
            assert not any(patch[1].cpu().tolist()) and not any(loss[1].cpu().tolist()) and all(not any(s.cpu().tolist()) for s in integrated[1]);assert loss[2].numel()==0
            actual=loss[0].cpu().numpy();assert exact(actual,integrated[0].cpu().numpy());raw=patch[0].cpu().numpy();scene_rows=[]
            refs=[r for r in prior['rows'] if r['scene']==scene and r['view']==view];score_refs=[r for r in scores['rows'] if r['scene']==scene and r['view']==view]
            assert exact(actual,np.array([r['score'] for r in score_refs],dtype=np.float64))
            np.savez_compressed(folder/'packed_rgb.npz',rgb=raw,rois=np.array(rois,dtype=np.int32),offsets=d['host']['prefix'][0],scores=actual)
            for e,en in enumerate(refs):
                path=V20/scene/f"parent_{en['parent_index']}"/en['appearance']/'query_patch.npz' if view=='fresh' else V23/scene/f'proposal_{e}_rgb_patch.npz'
                oldpatch=np.load(path);r=rois[e];assert r==oldpatch['bbox'].tolist();x0,y0,x1,y1=r;start,end=d['host']['prefix'][0][e:e+2];image=raw[int(start):int(end)].reshape(3,y1-y0,x1-x0)
                assert exact(image,oldpatch['rgb']);rebuilt=rgb.copy();rebuilt[:,y0:y1,x0:x1]=image;assert ahash(rebuilt)==en['rgb_sha256']
                row=dict(scene=scene,view=view,event=e,parent_id=en['parent_id'],appearance=en['appearance'],ROI=r,patch_literal=True,full_rgb_hash=ahash(rebuilt),
                    patch_hash=ahash(image),scores_literal_v24=True,score=actual[e].tolist(),native_forwards=forwards);rows.append(row);scene_rows.append(row)
                print(json.dumps(dict(scene=scene,view=view,event=e,packed_RGB_literal=True,loss_literal=True,native_forwards=forwards)),flush=True)
            assert initial==w.input_hashes();write_new(folder/'receipt.json',dict(rows=scene_rows,model_inputs_unchanged=True,native_forwards=forwards))
            del col,base,bg,br,bl,gt8,bs,cc,parents,small,cc_list,child,cg,boxes,offsets,d,patch,loss,integrated,raw;gc.collect();torch.cuda.empty_cache()
        del w,state;gc.collect();torch.cuda.empty_cache()
    assert forwards==p['new_native_forward_cap']==8 and len(rows)==32
    write_new(OUT/'results.json',dict(all_passed=True,rows=rows,new_native_forwards=forwards,training=False,accepted_mutations=0,
        no_complete_candidate_image_intermediate=True,loss_module_integrated=True,timing=False,source_sha256=sha(Path(__file__))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','build','fixtures','validate']);args=parser.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
