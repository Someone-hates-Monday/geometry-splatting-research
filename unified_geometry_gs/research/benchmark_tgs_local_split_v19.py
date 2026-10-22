"""Balanced paired cached-query timings after literal correctness gates; no optimizer."""
import argparse,gc,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,sha,write_new,module,pack,split,CrossWorkload
from trace_tgs_window_scene_v15 import call
from run_tgs_local_split_v19 import cache,load_local,same_bits
OUT=ROOT/'experiments/tgs_local_split_v19/timing';PARENT=OUT.parent;OLD=ROOT/'experiments/tgs_split_color_v14'

def telemetry():
    try:
        r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
        return dict(exit_code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
    except Exception as e:return dict(error=str(e))

def prepare():
    OUT.mkdir(exist_ok=True);r=json.loads((PARENT/'results.json').read_text());assert r['all_gates_passed'] and json.loads((PARENT/'independent_audit.json').read_text())['all_passed']
    paths=[Path(__file__),ROOT/'TGS_LOCAL_SPLIT_TIMING_PROTOCOL_V19.md',ROOT/'run_tgs_local_split_v19.py',ROOT/'tgs_local_split_v19.cu',
        ROOT/'tgs_split_color_common_v14.py',ROOT/'trace_tgs_window_scene_v15.py',PARENT/'plan.json',PARENT/'results.json',PARENT/'independent_audit.json',PARENT/'build_receipt.json']
    orders={}
    for row in r['rows']:
        key=f"{row['scene']}_c{row['camera']}_e{row['event']}";seed=int.from_bytes(hashlib.sha256(('v19cost/'+key).encode()).digest()[:8],'little')
        orders[key]=np.random.Generator(np.random.PCG64(seed)).permutation([0]*4+[1]*4).tolist()
    props=torch.cuda.get_device_properties(0)
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},orders=orders,cases=r['rows'],warmup_pairs=2,measured_pairs=8,
        full_forward_cap=644,torch=torch.__version__,cuda=torch.version.cuda,device=props.name,total_memory=props.total_memory,
        capability=list(torch.cuda.get_device_capability()),training=False))
    print('TIMING_PLAN_FROZEN',flush=True)

def measure(fn):
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();active=torch.cuda.memory_allocated()
    start=torch.cuda.Event(enable_timing=True);stop=torch.cuda.Event(enable_timing=True)
    start.record();t0=time.perf_counter_ns();out=fn();stop.record();torch.cuda.synchronize();t1=time.perf_counter_ns()
    result=dict(wall_ms=(t1-t0)/1e6,cuda_bracket_ms=start.elapsed_time(stop),peak_increment_bytes=torch.cuda.max_memory_allocated()-active)
    del out;return result

def storage_bytes(values):
    seen=set();total=0
    for v in values:
        if not isinstance(v,torch.Tensor):continue
        s=v.untyped_storage()
        if s.data_ptr() not in seen:seen.add(s.data_ptr());total+=s.nbytes()
    return total

@torch.no_grad()
def run():
    plan=json.loads((OUT/'plan.json').read_text());assert not (OUT/'results.json').exists()
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    original=json.loads((PARENT/'plan.json').read_text())
    for p,h in original['inputs'].items():assert sha(ROOT/p)==h,p
    native=module(original['native']['path']);m=load_local();rows=[];setups=[];forwards=0;before_telemetry=telemetry()
    for scene in ['flowers','room']:
        w=CrossWorkload(scene);initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh)
        for ci,cam in enumerate(w.cameras):
            col=torch.from_numpy(np.load(OLD/scene/f'baseline_{ci}/colors.npy')).cuda();torch.cuda.synchronize();t0=time.perf_counter_ns()
            base=call(native,state,cam,col);bg,br,bl,bpaths=cache(base,state,cam);alpha=base[2][1].contiguous();torch.cuda.synchronize();setup_ms=(time.perf_counter_ns()-t0)/1e6;forwards+=1
            setups.append(dict(scene=scene,camera=ci,setup_wall_ms=setup_ms,retained_native_plus_colors_bytes=storage_bytes(list(base)+[col,alpha]),pixels=cam['w']*cam['h']))
            for case in [x for x in plan['cases'] if x['scene']==scene and x['camera']==ci]:
                ei,index=case['event'],case['parent_id'];changed=split(state,index);child=pack(changed['points'][-4:],changed['sigma'][-4:],changed['opacity'][-4:],changed['sh'][-4:]);cc=col[index:index+1].repeat(4,1).contiguous();fullcol=torch.cat([col[:index],col[index+1:],cc]).contiguous()
                def full():return call(native,changed,cam,fullcol)
                def local():
                    small=call(native,child,cam,cc);cg,cr,cl,_=cache(small,child,cam)
                    result=m.query(bg,br,bl,col,cg,cr,cl,cc,index,base[1],alpha,bpaths,cam['w'],cam['h'])
                    return small,result
                for warm in range(2):
                    if warm==0:
                        f=full();q=local();torch.cuda.synchronize()
                        _,_,_,fp=cache(f,changed,cam)
                        assert same_bits(q[1][0],f[1]) and same_bits(q[1][1],f[2][1]) and same_bits(q[1][2],fp)
                        del f,q,fp
                    else:
                        q=local();torch.cuda.synchronize();del q
                        f=full();torch.cuda.synchronize();del f
                    forwards+=2
                key=f'{scene}_c{ci}_e{ei}';pairs=[]
                for rep,order in enumerate(plan['orders'][key]):
                    a,b=(('full',full),('local',local)) if order==0 else (('local',local),('full',full))
                    pair={a[0]:measure(a[1]),b[0]:measure(b[1])};forwards+=2
                    pairs.append(dict(repeat=rep,order='full_then_local' if order==0 else 'local_then_full',**pair,
                        full_over_local_wall=pair['full']['wall_ms']/pair['local']['wall_ms']))
                row=dict(scene=scene,camera=ci,event=ei,affected_tiles=case['affected_tiles'],affected_pixels=case['affected_pixels'],pixels=case['pixels'],pairs=pairs)
                rows.append(row);write_new(OUT/(key+'.json'),row)
                print(json.dumps(dict(case=key,median_full_over_local=float(np.median([p['full_over_local_wall'] for p in pairs])),native_forwards=forwards)),flush=True)
                del changed,child,cc,fullcol,full,local
            del base,bg,br,bl,bpaths,alpha,col
        assert initial==w.input_hashes()
        del w,state;gc.collect();torch.cuda.empty_cache()
    assert forwards==644 and len(rows)==32
    write_new(OUT/'results.json',dict(rows=rows,setups=setups,full_forwards=forwards,telemetry_before=before_telemetry,telemetry_after=telemetry(),
        training=False,scope='Prepared cached single-event query only; local includes native four-child forward and clones'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
