"""Matched FP64-score four-arm complete candidate endpoints; fixed32 measured blocks."""
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,write_new,module,pack,split,colors,CrossWorkload
from tgs_aux_ablation_common_v21 import control,view_list
from tgs_batch_score_common_v22 import backend as planner_backend,bank
from tgs_local_image_loss_common_v24 import backend as loss_backend,layout,device_layout,evaluate
from tgs_packed_query_common_v25 import OUT,backend,query
from run_tgs_local_split_v19 import cache,load_local
from trace_tgs_window_scene_v15 import call
from benchmark_tgs_local_split_v19 import measure,storage_bytes,telemetry
TIM=OUT/'timing';ARMS=['A_full_score','B_cached_image','C_sequential_packed','D_batch_packed']

def prepare():
    assert json.loads((OUT/'results.json').read_text())['all_passed'] and json.loads((OUT/'independent_audit.json').read_text())['all_passed'];TIM.mkdir()
    files=[Path(__file__),ROOT/'TGS_PACKED_QUERY_TIMING_PROTOCOL_V25.md',ROOT/'benchmark_tgs_local_split_v19.py',ROOT/'tgs_packed_query_common_v25.py',
        ROOT/'tgs_local_image_loss_common_v24.py',ROOT/'tgs_batch_score_common_v22.py',ROOT/'run_tgs_local_split_v19.py',ROOT/'trace_tgs_window_scene_v15.py',
        OUT/'plan.json',OUT/'runtime_dependency_addendum.json',OUT/'build_receipt.json',OUT/'results.json',OUT/'independent_audit.json',ROOT/'experiments/tgs_local_image_loss_v24/results.json']
    orders={};design=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]];views=[]
    for scene in ['flowers','room']:
        for view in ['fresh','old_0']:
            key=f'{scene}_{view}';seed=int.from_bytes(hashlib.sha256(('v25pipeline/'+key).encode()).digest()[:8],'little');rng=np.random.Generator(np.random.PCG64(seed));mapping=rng.permutation(4)
            orders[key]=[[ARMS[int(mapping[j])] for j in design[int(i)]] for i in rng.permutation([0,0,1,1,2,2,3,3])];views.append(dict(scene=scene,view=view,K=8))
    warm=2;repeats=8;cap=sum(1+(warm+repeats)*(3*v['K']+1) for v in views);assert cap==1004
    write_new(TIM/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},orders=orders,views=views,arms=ARMS,warm_blocks=warm,measured_blocks=repeats,
        derived_control_forward_cap=cap,score_atol=1e-13,score_rtol=1e-9,training=False,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()))
    print('PACKED_PIPELINE_TIMING_PLAN_FROZEN',flush=True)

@torch.no_grad()
def run():
    p=json.loads((TIM/'plan.json').read_text())
    for filename in [TIM/'plan.json',OUT/'plan.json',OUT/'runtime_dependency_addendum.json']:
        for n,h in json.loads(filename.read_text())['inputs'].items():assert sha(ROOT/n)==h,n
    assert not (TIM/'results.json').exists();write_new(TIM/'telemetry_start.json',telemetry())
    emitter=backend();losser=loss_backend();planner=planner_backend();ctrl=control();image_query=load_local();helper=module(HELPER);forwards=0;rows=[]
    cases_all=json.loads((OUT/'plan.json').read_text())['cases'];references=json.loads((ROOT/'experiments/tgs_local_image_loss_v24/results.json').read_text())['rows']
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh)
        for item in [v for v in p['views'] if v['scene']==scene]:
            view=item['view'];K=item['K'];key=f'{scene}_{view}';cam=next(v['cam'] for v in view_list(w,scene) if v['label']==view);W,H=cam['w'],cam['h'];col=colors(helper,state,cam)
            gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy' if view=='fresh' else ROOT/f'experiments/tgs_gt_quality_v11/{scene}/gt_rgb.npy');gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda()
            torch.cuda.synchronize();t0=time.perf_counter_ns();base=call(ctrl,state,cam,col);bg,br,bl,paths=cache(base,state,cam);alpha=base[2][1].contiguous();torch.cuda.synchronize();S=(time.perf_counter_ns()-t0)/1e6;forwards+=1
            setup=dict(scene=scene,view=view,setup_wall_ms=S,retained_cache_bytes_GT_excluded=storage_bytes(list(base)+[col,alpha]),GT_bytes=storage_bytes([gt8]),model_proposal_loading_excluded=True)
            write_new(TIM/(key+'_setup.json'),setup);print(json.dumps(dict(setup=key,wall_ms=S,native_forwards=forwards)),flush=True)
            cases=[r for r in cases_all if r['scene']==scene];assert len(cases)==K;bs,cc,parents,small,cc_list=bank(state,col,cam,cases,helper)
            changed=[split(state,r['parent_id']) for r in cases];fc=[torch.cat([col[:r['parent_id']],col[r['parent_id']+1:],c]).contiguous() for r,c in zip(cases,cc_list)]
            expected=np.array([r['score'] for r in references if r['scene']==scene and r['view']==view],dtype=np.float64);full_layout=device_layout(layout([[0,0,W,H]],W,H));one=[parents[e:e+1].contiguous() for e in range(K)]
            def full_score():
                scores=[];stats=[]
                for e in range(K):
                    f=call(ctrl,changed[e],cam,fc[e]);out=evaluate(losser,base[1],gt8,f[1].reshape(-1),full_layout,False);scores.append(out[0]);stats.append(out[1]);del f,out
                return torch.cat(scores),stats
            def cached_image():
                scores=[];stats=[]
                for e in range(K):
                    tiny=call(ctrl,small[e],cam,cc_list[e]);cg,cr,cl,_=cache(tiny,small[e],cam);q=image_query.query(bg,br,bl,col,cg,cr,cl,cc_list[e],cases[e]['parent_id'],base[1],alpha,paths,W,H)
                    out=evaluate(losser,base[1],gt8,q[0].reshape(-1),full_layout,False);scores.append(out[0]);stats.append(out[1]);del tiny,cg,cr,cl,q,out
                return torch.cat(scores),stats
            def sequential_packed():
                scores=[];stats=[]
                for e in range(K):
                    tiny=call(ctrl,small[e],cam,cc_list[e]);cg,_,_,_=cache(tiny,small[e],cam);q=query(emitter,losser,planner,bg,br,bl,col,cg,cc_list[e],one[e],base[1],gt8)
                    scores.append(q[0]);stats.extend(q[1]);del tiny,cg,q
                return torch.cat(scores),stats
            def batch_packed():
                tiny=call(ctrl,bs,cam,cc);cg,_,_,_=cache(tiny,bs,cam)
                return query(emitter,losser,planner,bg,br,bl,col,cg,cc,parents,base[1],gt8)
            funcs=dict(zip(ARMS,[full_score,cached_image,sequential_packed,batch_packed]))
            for warm,order in enumerate([ARMS,list(reversed(ARMS))]):
                for arm in order:
                    out=funcs[arm]();torch.cuda.synchronize();forwards+=1 if arm=='D_batch_packed' else K
                    if warm==0:
                        actual=out[0].cpu().numpy();assert actual.shape==expected.shape and np.all(np.abs(actual-expected)<=p['score_atol']+p['score_rtol']*np.abs(expected)),(key,arm)
                        assert all(not any(s.cpu().tolist()) for s in out[1])
                    del out
            blocks=[]
            for rep,order in enumerate(p['orders'][key]):
                measurements={}
                for arm in order:measurements[arm]=measure(funcs[arm]);forwards+=1 if arm=='D_batch_packed' else K
                ratios=dict(full_over_batch=measurements[ARMS[0]]['wall_ms']/measurements[ARMS[3]]['wall_ms'],cached_image_over_sequential=measurements[ARMS[1]]['wall_ms']/measurements[ARMS[2]]['wall_ms'],
                    sequential_over_batch=measurements[ARMS[2]]['wall_ms']/measurements[ARMS[3]]['wall_ms'],cached_image_over_batch=measurements[ARMS[1]]['wall_ms']/measurements[ARMS[3]]['wall_ms'])
                block=dict(repeat=rep,order=order,measurements=measurements,ratios=ratios,cumulative_native_forwards=forwards);blocks.append(block)
                write_new(TIM/(key+f'_block{rep}.json'),block)
            assert initial==w.input_hashes();row=dict(scene=scene,view=view,K=K,blocks=blocks,setup=setup,model_inputs_unchanged=True);write_new(TIM/(key+'.json'),row);rows.append(row)
            print(json.dumps(dict(view=key,median_full_over_batch=float(np.median([b['ratios']['full_over_batch'] for b in blocks])),native_forwards=forwards)),flush=True)
            del col,gt8,base,bg,br,bl,paths,alpha,bs,cc,parents,small,cc_list,changed,fc,full_layout,one,funcs,full_score,cached_image,sequential_packed,batch_packed;gc.collect();torch.cuda.empty_cache()
        del w,state;gc.collect();torch.cuda.empty_cache()
    assert forwards==p['derived_control_forward_cap'] and len(rows)==len(p['views'])
    for filename in [TIM/'plan.json',OUT/'plan.json',OUT/'runtime_dependency_addendum.json']:
        for n,h in json.loads(filename.read_text())['inputs'].items():assert sha(ROOT/n)==h,n
    write_new(TIM/'results.json',dict(rows=rows,native_forwards=forwards,training=False,measured_endpoints=4*8*4,FP64_oracle_not_FP32_training=True,telemetry_after=telemetry()))
    print('PACKED_PIPELINE_TIMING_COMPLETED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    try:globals()[args.action]()
    except BaseException:
        TIM.mkdir(exist_ok=True)
        with (TIM/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
