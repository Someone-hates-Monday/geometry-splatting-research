"""Unchanged v19 pipeline on large-coverage/new-color cases, no fallback."""
import argparse,gc,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from tgs_split_color_common_v14 import ROOT,HELPER,sha,write_new,module,pack,split,colors
from tgs_sh_isolation_common_v12 import workload
from trace_tgs_window_scene_v15 import call
from run_tgs_local_split_v19 import cache,load_local,same_bits
from benchmark_tgs_local_split_v19 import measure,storage_bytes,telemetry
OUT=ROOT/'experiments/tgs_local_stress_v20/timing';PARENT=OUT.parent

def prepare():
    OUT.mkdir(exist_ok=True);r=json.loads((PARENT/'results.json').read_text());assert r['all_gates_passed'] and json.loads((PARENT/'independent_audit.json').read_text())['all_passed']
    files=[Path(__file__),ROOT/'TGS_LOCAL_STRESS_TIMING_PROTOCOL_V20.md',ROOT/'benchmark_tgs_local_split_v19.py',
        ROOT/'run_tgs_local_split_v19.py',ROOT/'run_tgs_local_stress_v20.py',PARENT/'plan.json',PARENT/'results.json',PARENT/'independent_audit.json']
    files.extend(PARENT/scene/'selection.json' for scene in ['flowers','room'])
    orders={}
    for row in r['rows']:
        key=f"{row['scene']}_p{row['parent_index']}_{row['appearance']}";seed=int.from_bytes(hashlib.sha256(('v20cost/'+key).encode()).digest()[:8],'little')
        orders[key]=np.random.Generator(np.random.PCG64(seed)).permutation([0]*4+[1]*4).tolist()
    prop=torch.cuda.get_device_properties(0)
    write_new(OUT/'plan.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in files},cases=r['rows'],orders=orders,full_forward_cap=322,
        device=prop.name,total_memory=prop.total_memory,torch=torch.__version__,cuda=torch.version.cuda,training=False,routing=False))
    print('STRESS_TIMING_PLAN_FROZEN',flush=True)

@torch.no_grad()
def run():
    plan=json.loads((OUT/'plan.json').read_text());assert not (OUT/'results.json').exists()
    for p,h in plan['inputs'].items():assert sha(ROOT/p)==h,p
    original=json.loads((PARENT/'plan.json').read_text())
    for p,h in original['inputs'].items():assert sha(ROOT/p)==h,p
    for name in ['native','local']:assert sha(Path(original[name]['path']))==original[name]['sha256']
    native=module(original['native']['path']);m=load_local();helper=module(HELPER);rows=[];setups=[];forwards=0;before=telemetry()
    for scene in ['flowers','room']:
        w,cameras=workload(scene);ci=next(i for i,c in enumerate(cameras) if c['cohort']=='fresh');cam=w.cameras[ci];initial=w.input_hashes();state=pack(w.points,w.sigma,w.opacity,w.sh);col=colors(helper,state,cam)
        torch.cuda.synchronize();t0=time.perf_counter_ns();base=call(native,state,cam,col);bg,br,bl,bp=cache(base,state,cam);alpha=base[2][1].contiguous();torch.cuda.synchronize();S=(time.perf_counter_ns()-t0)/1e6;forwards+=1
        setups.append(dict(scene=scene,camera=cameras[ci]['camera']['img_name'],setup_wall_ms=S,retained_native_plus_colors_bytes=storage_bytes(list(base)+[col,alpha])))
        for case in [c for c in plan['cases'] if c['scene']==scene]:
            pi,index,appearance=case['parent_index'],case['parent_id'],case['appearance'];changed=split(state,index);child=pack(changed['points'][-4:],changed['sigma'][-4:],changed['opacity'][-4:],changed['sh'][-4:])
            cc=col[index:index+1].repeat(4,1).contiguous() if appearance=='hold' else colors(helper,child,cam);fc=torch.cat([col[:index],col[index+1:],cc]).contiguous()
            def full():return call(native,changed,cam,fc)
            def local():
                small=call(native,child,cam,cc);cg,cr,cl,_=cache(small,child,cam)
                return small,m.query(bg,br,bl,col,cg,cr,cl,cc,index,base[1],alpha,bp,cam['w'],cam['h'])
            for warm in range(2):
                if warm==0:
                    f=full();q=local();torch.cuda.synchronize();_,_,_,fp=cache(f,changed,cam)
                    assert same_bits(q[1][0],f[1]) and same_bits(q[1][1],f[2][1]) and same_bits(q[1][2],fp)
                    del f,q,fp
                else:
                    q=local();torch.cuda.synchronize();del q
                    f=full();torch.cuda.synchronize();del f
                forwards+=2
            key=f'{scene}_p{pi}_{appearance}';pairs=[]
            for rep,order in enumerate(plan['orders'][key]):
                a,b=(('full',full),('local',local)) if order==0 else (('local',local),('full',full))
                pair={a[0]:measure(a[1]),b[0]:measure(b[1])};forwards+=2
                pairs.append(dict(repeat=rep,order='full_then_local' if order==0 else 'local_then_full',**pair,full_over_local_wall=pair['full']['wall_ms']/pair['local']['wall_ms']))
            row=dict(scene=scene,parent_index=pi,parent_id=index,appearance=appearance,coverage=case['affected_pixels']/case['pixels'],
                candidate_work_fraction=case['candidate_pixel_visits_upper']/case['all_candidate_pixel_visits_upper'],pairs=pairs)
            rows.append(row);write_new(OUT/(key+'.json'),row)
            print(json.dumps(dict(case=key,median_full_over_local=float(np.median([p['full_over_local_wall'] for p in pairs])),coverage=row['coverage'],native_forwards=forwards)),flush=True)
            del changed,child,cc,fc,full,local
        assert initial==w.input_hashes();del w,state,col,base,bg,br,bl,bp,alpha;gc.collect();torch.cuda.empty_cache()
    assert forwards==322 and len(rows)==16
    write_new(OUT/'results.json',dict(rows=rows,setups=setups,full_forwards=forwards,telemetry_before=before,telemetry_after=telemetry(),training=False,routing=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);args=a.parse_args()
    try:globals()[args.action]()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        with (OUT/(args.action+'_failure.txt')).open('x') as f:f.write(traceback.format_exc())
        raise
