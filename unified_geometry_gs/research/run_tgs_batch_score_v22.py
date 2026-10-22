"""Freeze/build/check exact batched counterfactual PNG8 scores. No optimizer/training."""
import argparse,gc,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.cpp_extension import load
from tgs_split_color_common_v14 import ROOT,HELPER,sha,ahash,write_new,module,pack,split,colors,CrossWorkload,exact
from tgs_aux_ablation_common_v21 import V20,control,view_list
from run_tgs_local_split_v19 import cache,cache_gate,same_bits,fake
from trace_tgs_window_scene_v15 import call
from tgs_image_metrics_v11 import png8
from tgs_batch_score_common_v22 import OUT,backend,bank,schedule,cpu_sse
SOURCE=ROOT/'tgs_batch_score_v22.cu'

def prepare():
    OUT.mkdir(exist_ok=True);prior=json.loads((V20/'results.json').read_text());assert prior['all_gates_passed']
    files=[SOURCE,Path(__file__),ROOT/'TGS_BATCH_SCORE_PROTOCOL_V22.md',ROOT/'tgs_batch_score_common_v22.py',
        ROOT/'run_tgs_local_split_v19.py',ROOT/'tgs_aux_ablation_common_v21.py',ROOT/'tgs_split_color_common_v14.py',ROOT/'run_tgs_real_events.py',
        ROOT/'tgs_sh_isolation_common_v12.py',ROOT/'tgs_cross_scene_common_v10.py',ROOT/'trace_tgs_window_scene_v15.py',ROOT/'tgs_image_metrics_v11.py',HELPER,
        V20/'results.json',V20/'evidence_manifest.json',ROOT/'experiments/tgs_aux_ablation_v21/build_receipt.json',ROOT/'experiments/tgs_aux_ablation_v21/evidence_manifest.json',
        ROOT/'experiments/tgs_sh_isolation_v12/input_plan.json',ROOT/'experiments/tgs_sh_isolation_v12/identity_audit.json']
    old=json.loads((V20/'plan.json').read_text());files.extend(ROOT/p for p in old['inputs'] if p.endswith('point_cloud_state_dict.pt'))
    for scene in ['flowers','room']:
        files.extend([V20/scene/'selection.json',V20/scene/'baseline.npz',ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy'])
        files.extend(ROOT/f'sources/tgs_sh_isolation_v12/{scene}/fresh'/n for n in ['published_gt.png','original.JPG'])
        files.extend(V20/scene/f'parent_{i}/input.npz' for i in range(4))
    ctrl=json.loads((ROOT/'experiments/tgs_aux_ablation_v21/build_receipt.json').read_text());assert sha(Path(ctrl['path']))==ctrl['sha256']
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},cases=prior['rows'],control=ctrl,real_forward_cap=20,
        training=False,proposal_selection_uses_GT=False,score='Exact signed PNG8 SSE difference; not float L1+DSSIM'))
    print('BATCH_SCORE_PLAN_FROZEN',flush=True)

def frozen():
    p=json.loads((OUT/'plan.json').read_text())
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    assert sha(Path(p['control']['path']))==p['control']['sha256'];return p

def build():
    frozen();folder=OUT/'build';folder.mkdir();os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'];m=load(name='tgs_batch_score_v22',sources=[str(SOURCE)],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(folder),verbose=True)
    write_new(OUT/'build_receipt.json',dict(path=m.__file__,sha256=sha(Path(m.__file__)),source_sha256=sha(SOURCE),torch=torch.__version__,cuda=torch.version.cuda));return m

def score(m,bg,br,bl,col,cg,cc,parents,boxes,offsets,base8,gt8,base,alpha,paths,W,H,debug=-1):
    return m.score(bg,br,bl,col,cg,cc,parents,boxes,offsets,base8,gt8,base,alpha,paths,W,H,debug)

def composite(layers):
    T=np.float32(1);C=np.zeros(3,dtype=np.float32);last=0;median=-1
    for j,(a,color) in enumerate(layers):
        a=np.float32(a);test=np.float32(T*np.float32(1-a))
        if a<np.float32(1/255):continue
        if test<np.float32(1e-4):break
        if T>.5:median=j+1
        C=np.add(C,np.multiply(np.multiply(color,a,dtype=np.float32),T,dtype=np.float32),dtype=np.float32);T=test;last=j+1
    return C,np.float32(1-T),[last,median]

@torch.no_grad()
def fixtures():
    m=build();values=[np.float32(-.1),np.float32(0),np.float32(1),np.float32(1.1)]
    for k in range(255):
        x=np.float32((k+.5)/255);values.extend([np.nextafter(x,np.float32(-np.inf)),x,np.nextafter(x,np.float32(np.inf))])
    values=np.array(values,dtype=np.float32);expected=np.clip(np.add(np.multiply(values,np.float32(255),dtype=np.float32),np.float32(.5),dtype=np.float32),0,255).astype(np.uint8)
    assert np.array_equal(m.quantize(torch.from_numpy(values).cuda()).cpu().numpy(),expected)
    rows=[]
    for name in ['multi_ties_empty','all257']:
        N,K=(4,4) if name=='multi_ties_empty' else (258,1);bg=fake(N,[1.,2.,2.,3.] if N==4 else None,opacity=.5 if N==4 else 1/128)
        if N==4:bg[6][3]=0
        cg=fake(4*K,[2.]* (4*K));parents=torch.tensor([1,1,0,3] if K==4 else [0],device='cuda',dtype=torch.int32)
        if K==4:cg[6][8:]=0
        else:cg[6].zero_()
        oldids=[0,1,2] if N==4 else list(range(N));br=torch.tensor([0,len(oldids)],device='cuda',dtype=torch.int32);bl=torch.tensor(oldids,device='cuda',dtype=torch.int32)
        bc=torch.ones((N,3),device='cuda');cc=torch.ones((4*K,3),device='cuda')
        if K==4:
            bc[:3]=torch.eye(3,device='cuda');cc[:4]=torch.tensor([0.,1.,0.],device='cuda');cc[4:8]=torch.tensor([[.25,.5,.75],[.5,.25,.125],[.125,.25,.5],[.75,.5,.25]],device='cuda')
        old_layers=[(float(bg[4][i,3].item()),bc[i].cpu().numpy()) for i in oldids];C,A,P=composite(old_layers)
        base=torch.from_numpy(C.copy()).cuda().reshape(3,1,1);alpha=torch.tensor([[A]],device='cuda');paths=torch.tensor(P,device='cuda',dtype=torch.int32).reshape(2,1,1)
        gt=np.array([[[13,97,229]]],dtype=np.uint8);gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda();base8=m.quantize(base);boxes,offsets=schedule(m,bg,cg,parents,1,1);expected_scores=[]
        for e in range(K):
            parent=int(parents[e].item());merged=[(float(bg[0][i].item()),0,i) for i in oldids if i!=parent]
            merged+=[(2.,1,j) for j in range(4) if cg[6][4*e+j].item()>0];merged.sort()
            ls=[(float((bg if kind==0 else cg)[4][i if kind==0 else 4*e+i,3].item()),(bc[i] if kind==0 else cc[4*e+i]).cpu().numpy()) for _,kind,i in merged]
            c,a,p=composite(ls);delta=cpu_sse(png8(c.reshape(3,1,1)),gt)-cpu_sse(png8(C.reshape(3,1,1)),gt);expected_scores.append(delta)
            out=score(m,bg,br,bl,bc,cg,cc,parents,boxes,offsets,base8,gt8,base,alpha,paths,1,1,e);torch.cuda.synchronize();assert not any(out[1].cpu().tolist())
            assert np.array_equal(out[2][:,0,0].cpu().numpy(),c) and out[3].item()==float(a) and out[4][:,0,0].cpu().tolist()==p
            assert out[0][e].item()==delta
        out=score(m,bg,br,bl,bc,cg,cc,parents,boxes,offsets,base8,gt8,base,alpha,paths,1,1);torch.cuda.synchronize();assert out[0].cpu().tolist()==expected_scores
        assert all(x.numel()==0 for x in out[2:]);rows.append(dict(case=name,passed=True,proposals=K,scores=expected_scores))
    write_new(OUT/'fixtures.json',dict(all_passed=True,quantization_values=len(values),rows=rows,software_only=True));print('BATCH_FIXTURES_PASSED',flush=True)

@torch.no_grad()
def validate():
    plan=frozen();assert json.loads((OUT/'fixtures.json').read_text())['all_passed'];m=backend();ctrl=control();helper=module(HELPER);rows=[];forwards=0
    for scene in ['flowers','room']:
        folder=OUT/scene;folder.mkdir();w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh);cam=next(x['cam'] for x in view_list(w,scene) if x['label']=='fresh');W,H=cam['w'],cam['h'];col=colors(helper,state,cam)
        original=call(ctrl,state,cam,col);torch.cuda.synchronize();forwards+=1;bg,br,bl,paths=cache(original,state,cam);alpha=original[2][1].contiguous();base=original[1];old=np.load(V20/scene/'baseline.npz')
        assert exact(base.cpu().numpy(),old['rgb']) and exact(alpha.cpu().numpy(),old['alpha']) and exact(paths.cpu().numpy(),old['paths'])
        gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy');assert gt.dtype==np.uint8 and gt.shape==(H,W,3)
        for n in ['published_gt.png','original.JPG']:assert np.array_equal(np.asarray(Image.open(ROOT/f'sources/tgs_sh_isolation_v12/{scene}/fresh'/n).convert('RGB')),gt)
        gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda();base8=m.quantize(base);torch.cuda.synchronize();assert np.array_equal(base8.cpu().numpy().transpose(1,2,0),png8(base.cpu().numpy()))
        base_sse=cpu_sse(png8(base.cpu().numpy()),gt);cases=[r for r in plan['cases'] if r['scene']==scene];bs,cc,parents,small,colors_list=bank(state,col,cam,cases,helper)
        np.savez_compressed(folder/'bank_inputs.npz',parents=parents.cpu().numpy(),points=bs['points'].cpu().numpy(),sigma=bs['sigma'].cpu().numpy(),opacity=bs['opacity'].cpu().numpy(),colors=cc.cpu().numpy())
        batch=call(ctrl,bs,cam,cc);torch.cuda.synchronize();forwards+=1;cg,_,_,_=cache(batch,bs,cam);boxes,offsets=schedule(m,bg,cg,parents,W,H)
        allscores=score(m,bg,br,bl,col,cg,cc,parents,boxes,offsets,base8,gt8,base,alpha,paths,W,H);torch.cuda.synchronize();assert not any(allscores[1].cpu().tolist());assert all(x.numel()==0 for x in allscores[2:])
        expected=[];scene_rows=[]
        for e,case in enumerate(cases):
            index=case['parent_id'];changed=split(state,index);assert same_bits(changed['points'][-4:],small[e]['points']);fc=torch.cat([col[:index],col[index+1:],colors_list[e]]).contiguous()
            full=call(ctrl,changed,cam,fc);torch.cuda.synchronize();forwards+=1;fg,fr,fl,fp=cache(full,changed,cam);cache_gate(bg,[x[4*e:4*e+4] for x in cg],fg,index)
            prior=case['native_hashes'];full_rgb=full[1].cpu().numpy();assert ahash(full_rgb)==prior['rgb'] and ahash(full[2][1].cpu().numpy())==prior['alpha'] and ahash(fp.cpu().numpy())==prior['paths']
            native_delta=m.delta(full[1],base8,gt8);out=score(m,bg,br,bl,col,cg,cc,parents,boxes,offsets,base8,gt8,base,alpha,paths,W,H,e);stats=m.audit(bg,br,bl,cg,parents,e,fr,fl,len(state['points']),W,H);torch.cuda.synchronize()
            assert not any(out[1].cpu().tolist()) and not any(stats[:4].cpu().tolist())
            checks=dict(rgb=same_bits(out[2],full[1]),alpha=same_bits(out[3],full[2][1]),paths=same_bits(out[4],fp));assert all(checks.values()),(scene,e,checks)
            delta=cpu_sse(png8(full_rgb),gt)-base_sse;assert delta==native_delta.item()==out[0][e].item()==allscores[0][e].item();expected.append(delta)
            row=dict(scene=scene,event=e,parent_index=case['parent_index'],parent_id=index,appearance=case['appearance'],checks=checks,delta_SSE=delta,baseline_SSE=base_sse,
                batch_score=int(allscores[0][e].item()),full_gpu_score=int(native_delta.item()),all_tile_lists_literal=True,visible_cache_fields_literal=True,
                debug_hashes=dict(rgb=ahash(out[2].cpu().numpy()),alpha=ahash(out[3].cpu().numpy()),paths=ahash(out[4].cpu().numpy())),reference_hashes=prior,native_forwards=forwards)
            write_new(folder/f'proposal_{e}.json',row);rows.append(row);scene_rows.append(row);print(json.dumps(dict(scene=scene,event=e,score_exact=True,delta_SSE=delta,native_forwards=forwards)),flush=True)
            del changed,fc,full,fg,fr,fl,fp,native_delta,out,stats,full_rgb
        assert allscores[0].cpu().tolist()==expected
        write_new(folder/'receipt.json',dict(rows=scene_rows,GT_byte_identity=True,baseline_SSE=base_sse,scores=expected,
            box_tile_count=int(offsets[-1].item()),no_full_image_output_in_score_mode=True))
        assert initial==w.input_hashes()
        del w,state,col,original,bg,br,bl,paths,alpha,base,gt8,base8,bs,cc,parents,small,colors_list,batch,cg,boxes,offsets,allscores;gc.collect();torch.cuda.empty_cache()
    assert forwards==20 and len(rows)==16
    write_new(OUT/'results.json',dict(all_gates_passed=True,rows=rows,full_forwards=forwards,training=False,accepted_mutations=0,GT_based_selection=False));print('ALL16_BATCH_SCORES_EXACT',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','fixtures','validate']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
