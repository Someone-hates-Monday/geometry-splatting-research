"""Counterbalanced four-arm query attribution; identical selected outputs, no training."""
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,write_new,module,pack,split,colors,CrossWorkload
from trace_tgs_window_scene_v15 import call
from run_tgs_local_split_v19 import cache,same_bits,load_local
from benchmark_tgs_local_split_v19 import measure,storage_bytes,telemetry
from tgs_aux_ablation_common_v21 import OUT,V20,control,view_list
TIM=OUT/'timing';ARMS=['A_full_aux','B_full_rgb','C_local_auxchild','D_local_rgbchild']

def prepare():
    TIM.mkdir(exist_ok=True);validation=json.loads((OUT/'validation/results.json').read_text());assert validation['all_gates_passed'] and validation['full_forwards']==204
    stress=json.loads((V20/'results.json').read_text());assert stress['all_gates_passed']
    paths=[Path(__file__),ROOT/'TGS_AUX_ABLATION_TIMING_PROTOCOL_V21.md',ROOT/'benchmark_tgs_local_split_v19.py',ROOT/'tgs_aux_ablation_common_v21.py',
        ROOT/'run_tgs_local_split_v19.py',OUT/'plan.json',OUT/'build_receipt.json',OUT/'validation/plan.json',OUT/'validation/results.json',
        V20/'plan.json',V20/'results.json',V20/'independent_audit.json']
    paths.extend(V20/s/'selection.json' for s in ['flowers','room']);orders={};williams=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]
    for row in stress['rows']:
        key=f"{row['scene']}_p{row['parent_index']}_{row['appearance']}";seed=int.from_bytes(hashlib.sha256(('v21quad/'+key).encode()).digest()[:8],'little');rng=np.random.Generator(np.random.PCG64(seed));mapping=rng.permutation(4)
        orders[key]=[[ARMS[int(mapping[j])] for j in williams[int(i)]] for i in rng.permutation([0,0,1,1,2,2,3,3])]
    prop=torch.cuda.get_device_properties(0)
    write_new(TIM/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},cases=stress['rows'],orders=orders,arms=ARMS,primary_ratio='B_full_rgb / D_local_rgbchild',
        full_forward_cap=644,device=prop.name,torch=torch.__version__,cuda=torch.version.cuda,training=False))
    print('QUAD_TIMING_PLAN_FROZEN',flush=True)

def check_outputs(outputs,changed,cam):
    _,_,_,paths=cache(outputs['A_full_aux'],changed,cam);reference=[outputs['A_full_aux'][1],outputs['A_full_aux'][2][1],paths]
    for key in ARMS[1:]:
        if key=='B_full_rgb':
            _,_,_,p=cache(outputs[key],changed,cam);actual=[outputs[key][1],outputs[key][2][1],p]
        else:actual=outputs[key][1][:3]
        assert all(same_bits(a,b) for a,b in zip(reference,actual)),key

@torch.no_grad()
def run():
    plan=json.loads((TIM/'plan.json').read_text());assert not (TIM/'results.json').exists()
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    source=json.loads((OUT/'plan.json').read_text())
    for p,h in source['inputs'].items():assert sha(ROOT/p)==h,p
    native=module(source['native']['path']);rgb=control();m=load_local();helper=module(HELPER);rows=[];setups=[];forwards=0;before=telemetry()
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh);cam=next(v['cam'] for v in view_list(w,scene) if v['label']=='fresh');col=colors(helper,state,cam)
        bases=[];parsed=[]
        for backend,label in [(native,'A_original_cache'),(rgb,'B_control_cache')]:
            torch.cuda.synchronize();t0=time.perf_counter_ns();b=call(backend,state,cam,col);g,r,l,p=cache(b,state,cam);alpha=b[2][1].contiguous();torch.cuda.synchronize();S=(time.perf_counter_ns()-t0)/1e6;forwards+=1
            setups.append(dict(scene=scene,kind=label,setup_wall_ms=S,retained_native_plus_colors_bytes=storage_bytes(list(b)+[col,alpha])))
            bases.append(b);parsed.append((g,r,l,p,alpha))
        ga,ra,la,pa,aa=parsed[0];gb,rb,lb,pb,ab=parsed[1]
        for case in [r for r in plan['cases'] if r['scene']==scene]:
            pi,index,appearance=case['parent_index'],case['parent_id'],case['appearance'];changed=split(state,index);child=pack(changed['points'][-4:],changed['sigma'][-4:],changed['opacity'][-4:],changed['sh'][-4:])
            cc=col[index:index+1].repeat(4,1).contiguous() if appearance=='hold' else colors(helper,child,cam);fc=torch.cat([col[:index],col[index+1:],cc]).contiguous()
            def full_aux():return call(native,changed,cam,fc)
            def full_rgb():return call(rgb,changed,cam,fc)
            def local_aux():
                small=call(native,child,cam,cc);cg,cr,cl,_=cache(small,child,cam)
                return small,m.query(ga,ra,la,col,cg,cr,cl,cc,index,bases[0][1],aa,pa,cam['w'],cam['h'])
            def local_rgb():
                small=call(rgb,child,cam,cc);cg,cr,cl,_=cache(small,child,cam)
                return small,m.query(gb,rb,lb,col,cg,cr,cl,cc,index,bases[1][1],ab,pb,cam['w'],cam['h'])
            functions=dict(zip(ARMS,[full_aux,full_rgb,local_aux,local_rgb]))
            outputs={k:functions[k]() for k in ARMS};torch.cuda.synchronize();check_outputs(outputs,changed,cam);forwards+=4;del outputs
            for k in reversed(ARMS):out=functions[k]();torch.cuda.synchronize();forwards+=1;del out
            key=f'{scene}_p{pi}_{appearance}';blocks=[]
            for rep,order in enumerate(plan['orders'][key]):
                values={}
                for k in order:values[k]=measure(functions[k]);forwards+=1
                ratios=dict(matched_full_over_local=values['B_full_rgb']['wall_ms']/values['D_local_rgbchild']['wall_ms'],
                    native_full_over_control=values['A_full_aux']['wall_ms']/values['B_full_rgb']['wall_ms'],
                    local_originalchild_over_controlchild=values['C_local_auxchild']['wall_ms']/values['D_local_rgbchild']['wall_ms'],
                    original_full_over_original_local=values['A_full_aux']['wall_ms']/values['C_local_auxchild']['wall_ms'])
                blocks.append(dict(repeat=rep,order=order,measurements=values,ratios=ratios))
            row=dict(scene=scene,parent_index=pi,parent_id=index,appearance=appearance,coverage=case['affected_pixels']/case['pixels'],blocks=blocks)
            write_new(TIM/(key+'.json'),row);rows.append(row);print(json.dumps(dict(case=key,matched_median=float(np.median([b['ratios']['matched_full_over_local'] for b in blocks])),native_forwards=forwards)),flush=True)
            del changed,child,cc,fc,functions,full_aux,full_rgb,local_aux,local_rgb
        assert initial==w.input_hashes()
        del w,state,col,bases,parsed,b,g,r,l,p,alpha,ga,ra,la,pa,aa,gb,rb,lb,pb,ab;gc.collect();torch.cuda.empty_cache()
    assert forwards==644 and len(rows)==16
    write_new(TIM/'results.json',dict(rows=rows,setups=setups,full_forwards=forwards,telemetry_before=before,telemetry_after=telemetry(),training=False,
        primary_scope='Prepared cached query vs auxiliary-arithmetic-ablated full rendering; API allocations retained'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        TIM.mkdir(exist_ok=True)
        with (TIM/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
