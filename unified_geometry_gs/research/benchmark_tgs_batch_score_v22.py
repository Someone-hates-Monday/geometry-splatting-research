"""Balanced same-score four-arm prepared-query timing; no policy,optimizer or training."""
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,write_new,module,pack,split,colors,CrossWorkload
from tgs_aux_ablation_common_v21 import control,view_list
from run_tgs_local_split_v19 import cache,load_local
from benchmark_tgs_local_split_v19 import measure,storage_bytes,telemetry
from trace_tgs_window_scene_v15 import call
from tgs_batch_score_common_v22 import OUT,backend,bank,schedule
from run_tgs_batch_score_v22 import score
TIM=OUT/'timing';ARMS=['A_full_score','B_cached_image_score','C_sequential_scalar','D_batch_scalar']

def prepare():
    TIM.mkdir(exist_ok=True);r=json.loads((OUT/'results.json').read_text());assert r['all_gates_passed'] and json.loads((OUT/'independent_audit.json').read_text())['all_passed']
    paths=[Path(__file__),ROOT/'TGS_BATCH_SCORE_TIMING_PROTOCOL_V22.md',ROOT/'benchmark_tgs_local_split_v19.py',
        ROOT/'tgs_batch_score_common_v22.py',ROOT/'run_tgs_batch_score_v22.py',ROOT/'tgs_aux_ablation_common_v21.py',
        ROOT/'run_tgs_local_split_v19.py',OUT/'plan.json',OUT/'results.json',OUT/'independent_audit.json',OUT/'build_receipt.json']
    orders={};design=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]
    for scene in ['flowers','room']:
        for K in [1,2,4,8]:
            key=f'{scene}_K{K}';seed=int.from_bytes(hashlib.sha256(('v22score/'+key).encode()).digest()[:8],'little');rng=np.random.Generator(np.random.PCG64(seed));mapping=rng.permutation(4)
            orders[key]=[[ARMS[int(mapping[j])] for j in design[int(i)]] for i in rng.permutation([0,0,1,1,2,2,3,3])]
    write_new(TIM/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},orders=orders,sizes=[1,2,4,8],arms=ARMS,full_forward_cap=954,
        primary='A_full_score / D_batch_scalar for K8',device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,training=False))
    print('BATCH_SCORE_TIMING_PLAN_FROZEN',flush=True)

@torch.no_grad()
def run():
    p=json.loads((TIM/'plan.json').read_text());assert not (TIM/'results.json').exists()
    for n,h in p['inputs'].items():assert sha(ROOT/n)==h,n
    source=json.loads((OUT/'plan.json').read_text())
    for n,h in source['inputs'].items():assert sha(ROOT/n)==h,n
    ctrl=control();m=backend();image_query=load_local();helper=module(HELPER);rows=[];setups=[];forwards=0;before=telemetry()
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh);cam=next(v['cam'] for v in view_list(w,scene) if v['label']=='fresh');W,H=cam['w'],cam['h'];col=colors(helper,state,cam)
        gt=np.load(ROOT/f'experiments/tgs_sh_isolation_v12/{scene}/fresh/gt_rgb.npy');gt8=torch.from_numpy(gt.transpose(2,0,1).copy()).cuda();torch.cuda.synchronize();t0=time.perf_counter_ns()
        base=call(ctrl,state,cam,col);bg,br,bl,paths=cache(base,state,cam);alpha=base[2][1].contiguous();base8=m.quantize(base[1]);torch.cuda.synchronize();S=(time.perf_counter_ns()-t0)/1e6;forwards+=1
        setups.append(dict(scene=scene,setup_wall_ms=S,retained_cache_PNG_GT_bytes=storage_bytes(list(base)+[col,alpha,base8,gt8]),model_and_GT_loading_excluded=True))
        cases=[r for r in source['cases'] if r['scene']==scene];bs,cc,parents,small,cc_list=bank(state,col,cam,cases,helper)
        references=[r['delta_SSE'] for r in json.loads((OUT/'results.json').read_text())['rows'] if r['scene']==scene]
        changed=[split(state,r['parent_id']) for r in cases];fc=[torch.cat([col[:r['parent_id']],col[r['parent_id']+1:],c]).contiguous() for r,c in zip(cases,cc_list)]
        for K in p['sizes']:
            bank_state=pack(*[bs[key][:4*K] for key in ['points','sigma','opacity','sh']]);bank_color=cc[:4*K].contiguous();ps=parents[:K].contiguous()
            def full_score():
                scores=[]
                for e in range(K):
                    full=call(ctrl,changed[e],cam,fc[e]);scores.append(m.delta(full[1],base8,gt8));del full
                return torch.cat(scores),[]
            def cached_image_score():
                scores=[]
                for e in range(K):
                    tiny=call(ctrl,small[e],cam,cc_list[e]);cg,cr,cl,_=cache(tiny,small[e],cam)
                    q=image_query.query(bg,br,bl,col,cg,cr,cl,cc_list[e],cases[e]['parent_id'],base[1],alpha,paths,W,H)
                    scores.append(m.delta(q[0],base8,gt8));del q,tiny,cg,cr,cl
                return torch.cat(scores),[]
            def sequential_scalar():
                scores=[];stats=[]
                for e in range(K):
                    tiny=call(ctrl,small[e],cam,cc_list[e]);cg,_,_,_=cache(tiny,small[e],cam);one=parents[e:e+1];boxes,offsets=schedule(m,bg,cg,one,W,H)
                    out=score(m,bg,br,bl,col,cg,cc_list[e],one,boxes,offsets,base8,gt8,base[1],alpha,paths,W,H)
                    scores.append(out[0]);stats.append(out[1]);del tiny,cg,boxes,offsets,out
                return torch.cat(scores),stats
            def batch_scalar():
                tiny=call(ctrl,bank_state,cam,bank_color);cg,_,_,_=cache(tiny,bank_state,cam);boxes,offsets=schedule(m,bg,cg,ps,W,H)
                out=score(m,bg,br,bl,col,cg,bank_color,ps,boxes,offsets,base8,gt8,base[1],alpha,paths,W,H)
                return out[0],[out[1]]
            funcs=dict(zip(ARMS,[full_score,cached_image_score,sequential_scalar,batch_scalar]))
            for warm,order in enumerate([ARMS,list(reversed(ARMS))]):
                for method in order:
                    out=funcs[method]();torch.cuda.synchronize();forwards+=1 if method=='D_batch_scalar' else K
                    if warm==0:
                        assert out[0].cpu().tolist()==references[:K],(scene,K,method)
                        assert all(not any(s.cpu().tolist()) for s in out[1])
                    del out
            key=f'{scene}_K{K}';blocks=[]
            for rep,order in enumerate(p['orders'][key]):
                measured={}
                for method in order:measured[method]=measure(funcs[method]);forwards+=1 if method=='D_batch_scalar' else K
                ratios=dict(full_over_batch=measured['A_full_score']['wall_ms']/measured['D_batch_scalar']['wall_ms'],
                    cached_image_over_sequential_scalar=measured['B_cached_image_score']['wall_ms']/measured['C_sequential_scalar']['wall_ms'],
                    sequential_scalar_over_batch=measured['C_sequential_scalar']['wall_ms']/measured['D_batch_scalar']['wall_ms'],
                    cached_image_over_batch=measured['B_cached_image_score']['wall_ms']/measured['D_batch_scalar']['wall_ms'])
                blocks.append(dict(repeat=rep,order=order,measurements=measured,ratios=ratios))
            row=dict(scene=scene,K=K,proposals=[dict(parent_id=x['parent_id'],appearance=x['appearance'],coverage=x['affected_pixels']/x['pixels']) for x in cases[:K]],blocks=blocks)
            write_new(TIM/(key+'.json'),row);rows.append(row);print(json.dumps(dict(case=key,full_over_batch_median=float(np.median([b['ratios']['full_over_batch'] for b in blocks])),native_forwards=forwards)),flush=True)
            del bank_state,bank_color,ps,funcs,full_score,cached_image_score,sequential_scalar,batch_scalar
        assert initial==w.input_hashes()
        del w,state,col,gt8,base,bg,br,bl,paths,alpha,base8,bs,cc,parents,small,cc_list,changed,fc;gc.collect();torch.cuda.empty_cache()
    assert forwards==954 and len(rows)==8
    write_new(TIM/'results.json',dict(rows=rows,setups=setups,full_forwards=forwards,telemetry_before=before,telemetry_after=telemetry(),training=False,
        provided_baseline_PNG_and_GT_in_warm_context=True,policy_selection=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        TIM.mkdir(exist_ok=True)
        with (TIM/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
