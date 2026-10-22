"""All48 prior edits and six baselines must retain literal selected outputs."""
import gc,json,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,ahash,write_new,module,pack,split,colors,CrossWorkload
from tgs_sh_isolation_common_v12 import capture
from trace_tgs_window_scene_v15 import call
from run_tgs_local_split_v19 import cache,cache_gate,same_bits,load_local
from tgs_aux_ablation_common_v21 import OUT,V19,V20,V14,control,view_list,rgba_equal
VAL=OUT/'validation'

def check_pair(name,a,b,state,cam):
    try:return rgba_equal(a,b,state,state,cam)
    except BaseException:
        _,_,_,ap=cache(a,state,cam);_,_,_,bp=cache(b,state,cam)
        np.savez_compressed(VAL/(name+'_mismatch.npz'),original_rgb=a[1].cpu().numpy(),control_rgb=b[1].cpu().numpy(),
            original_alpha=a[2][1].cpu().numpy(),control_alpha=b[2][1].cpu().numpy(),original_paths=ap.cpu().numpy(),control_paths=bp.cpu().numpy())
        raise

def aux_zero(res):
    assert bool(torch.all(res[2][[0,2,3,4,5,6]]==0)) and bool(torch.all(res[9]==0))

@torch.no_grad()
def main():
    VAL.mkdir(exist_ok=True);plan=json.loads((OUT/'plan.json').read_text());assert not (VAL/'results.json').exists()
    for n,h in plan['inputs'].items():assert sha(ROOT/n)==h,n
    write_new(VAL/'plan.json',dict(script_sha256=sha(Path(__file__)),source_plan_sha256=sha(OUT/'plan.json'),build_receipt_sha256=sha(OUT/'build_receipt.json'),
        original_forward_cap=204,cases=48,training=False))
    native=module(plan['native']['path']);rgb=control();query=load_local();helper=module(HELPER);rows=[];baselines=[];forwards=0
    oldresults=json.loads((V14/'results.json').read_text());freshresults=json.loads((ROOT/'experiments/tgs_coupled_v18/fresh/results.json').read_text())
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh)
        for view in view_list(w,scene):
            label,cam=view['label'],view['cam'];W,H=cam['w'],cam['h'];tiles=((W+15)//16)*((H+15)//16)
            col=torch.from_numpy(np.load(V14/scene/f"baseline_{view['ci']}/colors.npy")).cuda() if view['ci'] is not None else colors(helper,state,cam)
            a=call(native,state,cam,col);b=call(rgb,state,cam,col);torch.cuda.synchronize();forwards+=2
            arrays=capture(a,W,H)
            if view['ci'] is not None:
                prior=json.loads((V14/scene/f"baseline_{view['ci']}/receipt.json").read_text())['arrays']
                for k,v in arrays.items():assert ahash(v)==prior[k]['sha256']
            else:
                prior=next(r for r in freshresults['baselines'] if r['scene']==scene)['hashes']
                for k,v in arrays.items():assert ahash(v)==prior[k]
            del arrays
            checks=check_pair(scene+'_'+label+'_baseline',a,b,state,cam);aux_zero(b)
            baselines.append(dict(scene=scene,view=label,checks=checks,control_aux_zero=True))
            ga,ra,la,pa=cache(a,state,cam);gb,rb,lb,pb=cache(b,state,cam);aa=a[2][1].contiguous();ab=b[2][1].contiguous()
            for case in [r for r in plan['cases'] if r['scene']==scene and r['view']==label]:
                key=case['key'];index=case['parent_id'];changed=split(state,index);child=pack(changed['points'][-4:],changed['sigma'][-4:],changed['opacity'][-4:],changed['sh'][-4:])
                cc=col[index:index+1].repeat(4,1).contiguous() if case['appearance']=='hold' else colors(helper,child,cam);fc=torch.cat([col[:index],col[index+1:],cc]).contiguous()
                sa=call(native,child,cam,cc);sb=call(rgb,child,cam,cc);torch.cuda.synchronize();forwards+=2
                check_pair(key+'_child',sa,sb,child,cam);aux_zero(sb)
                ca,cra,cla,_=cache(sa,child,cam);cb,crb,clb,_=cache(sb,child,cam)
                qa=query.query(ga,ra,la,col,ca,cra,cla,cc,index,a[1],aa,pa,W,H);qb=query.query(gb,rb,lb,col,cb,crb,clb,cc,index,b[1],ab,pb,W,H)
                fa=call(native,changed,cam,fc);fb=call(rgb,changed,cam,fc);torch.cuda.synchronize();forwards+=2
                fa_arrays=capture(fa,W,H)
                if view['ci'] is not None:
                    old=next(r for r in oldresults['cases'] if r['scene']==scene and r['camera']==view['ci'] and r['event']==case['event'])
                    for k,v in fa_arrays.items():assert ahash(v)==old['hold_hashes'][k]['sha256']
                else:
                    expected=case['prior']['native_hashes'];assert ahash(fa_arrays['rgb'])==expected['rgb'] and ahash(fa_arrays['aux'][1])==expected['alpha'] and ahash(fa_arrays['contributors'].view(np.int32))==expected['paths']
                matched=check_pair(key+'_full',fa,fb,changed,cam);aux_zero(fb)
                fga,fra,fla,fpa=cache(fa,changed,cam);fgb,frb,flb,fpb=cache(fb,changed,cam);cache_gate(ga,ca,fga,index);cache_gate(gb,cb,fgb,index)
                ta=query.audit(ga,ra,la,ca,cra,cla,fra,fla,index,len(state['points']),tiles);tb=query.audit(gb,rb,lb,cb,crb,clb,frb,flb,index,len(state['points']),tiles);torch.cuda.synchronize()
                assert not any(ta[:4].cpu().tolist()) and not any(tb[:4].cpu().tolist())
                localchecks={}
                for name,q in [('original_child',qa),('control_child',qb)]:
                    localchecks[name]=dict(rgb=same_bits(q[0],fa[1]),alpha=same_bits(q[1],fa[2][1]),paths=same_bits(q[2],fpa));assert all(localchecks[name].values()),(key,name,localchecks[name])
                row=dict(key=key,scene=scene,view=label,parent_id=index,appearance=case['appearance'],full_checks=matched,local_checks=localchecks,
                    visible_geometry_and_tile_lists_literal=True,control_aux_zero=True,RGB_sha256=ahash(fb[1].cpu().numpy()),native_RGB_sha256=ahash(fa_arrays['rgb']),native_forwards=forwards)
                write_new(VAL/(key+'.json'),row);rows.append(row);print(json.dumps(dict(case=key,literal=True,native_forwards=forwards)),flush=True)
                del changed,child,cc,fc,sa,sb,ca,cra,cla,cb,crb,clb,qa,qb,fa,fb,fa_arrays,fga,fra,fla,fpa,fgb,frb,flb,fpb,ta,tb
            del a,b,ga,ra,la,pa,gb,rb,lb,pb,aa,ab,col
        assert initial==w.input_hashes();del w,state;gc.collect();torch.cuda.empty_cache()
    assert forwards==204 and len(rows)==48 and len(baselines)==6
    write_new(VAL/'results.json',dict(all_gates_passed=True,rows=rows,baselines=baselines,full_forwards=forwards,training=False,backward_exercised=False))
    print('AUX_CONTROL_ALL48_LITERAL',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        VAL.mkdir(exist_ok=True)
        with (VAL/'failure.txt').open('x') as f:f.write(traceback.format_exc())
        raise
