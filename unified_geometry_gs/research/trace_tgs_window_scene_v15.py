"""Three recaptures and a bounded exact-endpoint observer of all frozen violation pixels."""
import argparse,gc,json,os,sys,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load
from tgs_split_color_common_v14 import ROOT,sha,ahash,write_new,module,pack,split,CrossWorkload,exact
from tgs_sh_isolation_common_v12 import capture
OUT=ROOT/'experiments/tgs_window_scene_v15/trace';V15=OUT.parent;OLD=ROOT/'experiments/tgs_split_color_v14'
SOURCE=ROOT/'tgs_window_scene_trace_v15.cu'

def align(n):return (n+127)//128*128

def prepare():
    OUT.mkdir(exist_ok=True);target=OUT/'plan.json';assert not target.exists()
    oldplan=json.loads((OLD/'plan.json').read_text());pixels=json.loads((V15/'violation_pixels.json').read_text())
    assert len(pixels['pixels'])==3
    paths=[V15/'violation_pixels.json',V15/'independent_audit.json',OLD/'plan.json',OLD/'results.json',
        ROOT/'TGS_WINDOW_TRACE_ADDENDUM_V15.md',SOURCE,Path(__file__),
        ROOT/'sources/tgs_native_diagnostic_v1/native/cuda_rasterizer/forward.cu',
        ROOT/'sources/tgs_native_diagnostic_v1/native/cuda_rasterizer/rasterizer_impl.cu']
    for case in pixels['pixels']:
        base=OLD/case['scene'];paths.extend([base/f"event_{case['event']}/input.npz",base/f"baseline_{case['camera']}/colors.npy"])
    en=oldplan['modules']['native'];assert sha(Path(en['path']))==en['sha256']
    write_new(target,dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},module=en,pixels=pixels['pixels'],
        full_forward_cap=3,trace_cap=18,flags=['-O3','--use_fast_math','sm_120'],
        endpoint_gate='literal RGB,alpha,last ordinal at all6 endpoints',posthoc_mechanism=True,training=False))
    print(json.dumps(dict(trace_plan_frozen=True,pixels=len(pixels['pixels']))),flush=True)

def build():
    directory=OUT/'build';directory.mkdir()
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    mod=load(name='tgs_window_scene_trace_v15',sources=[str(SOURCE)],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(directory),verbose=True)
    write_new(OUT/'build_receipt.json',dict(path=mod.__file__,sha256=sha(Path(mod.__file__)),source_sha256=sha(SOURCE),
        script_sha256=sha(Path(__file__)),torch=torch.__version__,cuda=torch.version.cuda))
    return mod

def call(mod,state,cam,col):
    empty=torch.empty(0,device='cuda');return mod.rasterize_triangles(torch.zeros(3,device='cuda'),state['points'],state['sigma'],
        state['counts'],state['offsets'],col,state['opacity'],state['zero'].clone(),state['zero'].clone(),cam['view'],cam['projection'],
        len(state['points']),cam['tx'],cam['ty'],cam['h'],cam['w'],empty,3,cam['center'],False,False)

def gather(res,state,cam,col,x,y):
    n=len(state['points']);hw=cam['w']*cam['h'];tile=(y//16)*((cam['w']+15)//16)+x//16
    off=align(align(12*hw)+8*hw)
    start,end=res[6].narrow(0,off+8*tile,8).view(torch.int32).cpu().tolist()
    ids=res[5].narrow(0,4*start,4*(end-start)).view(torch.int32).long();raw=res[4]
    assert torch.all(res[3].index_select(0,ids)>0)
    pos=0;fields={}
    for name,size in [('depth',4),('clamped',3),('radii',4),('anchor',8),('conic',16),('phi',8),('rgb',12)]:
        pos=align(pos);fields[name]=pos;pos+=size*n
    maxoff=raw.numel()-128-8*n;minoff=maxoff-align(8*n);pw=minoff-align(12*n)
    normals=pw-align(24*n);offsets=normals-align(12*n)
    def get(offset,width):return raw.narrow(0,offset,4*n*width).view(torch.float32).reshape(n,width).index_select(0,ids)
    packed=torch.cat([get(normals,6),get(offsets,3),get(fields['phi'],2)[:,:1],state['sigma'].reshape(n,1).index_select(0,ids),
        get(fields['conic'],4)[:,3:],get(fields['depth'],1),col.index_select(0,ids),ids.float()[:,None]],1).contiguous()
    imageoff=align(12*hw);last=res[6].narrow(0,imageoff+4*(y*cam['w']+x),4).view(torch.int32).cpu().item()
    endpoint=torch.cat([res[1][:,y,x],res[2][1,y,x].reshape(1),torch.tensor([last],device='cuda')]).cpu().numpy()
    return packed.cpu().numpy(),ids.cpu().numpy(),endpoint,dict(tile=tile,start=start,end=end)

@torch.no_grad()
def run():
    plan=json.loads((OUT/'plan.json').read_text());assert not (OUT/'results.json').exists()
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h
    native=module(plan['module']['path']);observer=build();w=CrossWorkload('room');before_hashes=w.input_hashes()
    state=pack(w.points,w.sigma,w.opacity,w.sh);cam=w.cameras[1]
    col=torch.from_numpy(np.load(OLD/'room/baseline_1/colors.npy')).cuda();allpackets={};forwards=0
    prior=json.loads((OLD/'results.json').read_text())
    for event in [None,0,5]:
        if event is None:
            current=state;cc=col;label='before';points=plan['pixels'];expected=json.loads((OLD/'room/baseline_1/receipt.json').read_text())['arrays']
        else:
            case=next(p for p in plan['pixels'] if p['event']==event);index=case['parent_id'];current=split(state,index)
            cc=torch.cat([col[:index],col[index+1:],col[index:index+1].repeat(4,1)]).contiguous();label=f'event_{event}'
            points=[p for p in plan['pixels'] if p['event']==event]
            expected=next(r for r in prior['cases'] if r['scene']=='room' and r['camera']==1 and r['event']==event)['hold_hashes']
        res=call(native,current,cam,cc);torch.cuda.synchronize();forwards+=1
        arrays=capture(res,cam['w'],cam['h'])
        for key,value in arrays.items():assert ahash(value)==expected[key]['sha256']
        for p in points:
            x,y=p['pixel'];packet,ids,endpoint,meta=gather(res,current,cam,cc,x,y)
            name=f'{label}_{x}_{y}';np.savez_compressed(OUT/(name+'_packet.npz'),packed=packet,ids=ids,native=endpoint)
            allpackets[(event,x,y)]=(packet,ids,endpoint,meta)
        del arrays,res
        if event is not None:del current,cc
        print(json.dumps(dict(stage='native_recapture',event=event,all_fields_literal=True,forwards=forwards)),flush=True)
    assert forwards==3
    rows=[];trace_calls=0
    def replay(label,packet,x,y,no_stop):
        nonlocal trace_calls
        tensor=torch.from_numpy(packet.copy()).cuda();trace,summary=observer.replay(tensor,x,y,no_stop);torch.cuda.synchronize();trace_calls+=1
        trace=trace.cpu().numpy();summary=summary.cpu().numpy();evaluated=trace[np.isfinite(trace[:,0])]
        np.savez_compressed(OUT/(label+'.npz'),packed=packet,trace=trace,summary=summary)
        return summary,evaluated
    endpoints=[]
    for p in plan['pixels']:
        x,y=p['pixel'];event=p['event'];old,oldids,oldnative,_=allpackets[(None,x,y)];new,newids,newnative,_=allpackets[(event,x,y)]
        for label,packet,native_ep in [('before',old,oldnative),('after',new,newnative)]:
            s,tr=replay(f'e{event}_{x}_{y}_{label}_actual',packet,x,y,False)
            equal=exact(s[:5],native_ep.astype(np.float32));endpoints.append(equal)
            rows.append(dict(event=event,pixel=[x,y],mode=label+'_actual',summary=s.tolist(),literal_endpoint=equal,evaluated=len(tr)))
    write_new(OUT/'endpoint_gate.json',dict(all_literal=all(endpoints),endpoints=len(endpoints),rows=rows))
    assert len(endpoints)==6 and all(endpoints)
    for p in plan['pixels']:
        x,y=p['pixel'];event=p['event'];parent=p['parent_id'];old,oldids,_,_=allpackets[(None,x,y)];new,newids,_,_=allpackets[(event,x,y)]
        N=len(state['points']);is_child=newids>=N-1
        canonical=np.where(newids<parent,newids,newids+1)
        assert np.array_equal(oldids[oldids!=parent],canonical[~is_child])
        oldpos=int(np.flatnonzero(oldids==parent)[0]);external=new[~is_child];child=new[is_child]
        grouped=np.concatenate([external[:oldpos],child,external[oldpos:]],axis=0)
        for name,packet,no_stop in [('before_nostop',old,True),('after_nostop',new,True),('after_group',grouped,False),('after_group_nostop',grouped,True)]:
            s,tr=replay(f'e{event}_{x}_{y}_{name}',packet,x,y,no_stop)
            rows.append(dict(event=event,pixel=[x,y],mode=name,summary=s.tolist(),evaluated=len(tr)))
    assert trace_calls==18 and w.input_hashes()==before_hashes
    write_new(OUT/'results.json',dict(rows=rows,full_forwards=forwards,trace_calls=trace_calls,
        all_native_fields_literal=True,all_primary_endpoints_literal=True,
        plan_sha256=sha(OUT/'plan.json'),script_sha256=sha(Path(__file__)),training=False,
        scope='All three posthoc mechanism pixels; diagnostic group order and no-stop, not renderer improvement'))
    print(json.dumps(dict(complete=True,full_forwards=forwards,trace_calls=trace_calls,endpoints_literal=True)),flush=True)

if __name__=='__main__':
    try:
        a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
        prepare() if args.action=='prepare' else run()
    except BaseException:
        OUT.mkdir(exist_ok=True);(OUT/'failure.txt').write_text(traceback.format_exc());raise
