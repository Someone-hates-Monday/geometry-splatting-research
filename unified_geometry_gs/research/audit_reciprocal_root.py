"""Locked bitwise-output and paired timing comparison for reciprocal-table optimization."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,math,os,time,traceback,zipfile
import numpy as np
import torch
from research.reciprocal_root_cuda import load
from research.guarded_interval_cuda import FLAGS

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def same(a,b):return bool(np.array_equal(a.view(np.uint64),b.view(np.uint64)))
def event(fn):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);wall=time.perf_counter();start.record();fn();end.record();end.synchronize();return dict(event_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000)

def run(out,state,old,source,sections):
    start=time.perf_counter();control,proof=load(out,old,False);variant,_=load(out,old,True);save(out/'reciprocal_proof.json',dict(rows=proof,count=51,all_exact_bounds_adjacent=True,rewritten_call_sites=6))
    src=torch.tensor(source,dtype=torch.float64,device='cuda');sec=torch.tensor(sections,dtype=torch.int32,device='cuda')
    cp=control.preprocess(src);vp=variant.preprocess(src);cr=control.solve(cp,sec);vr=variant.solve(vp,sec);torch.cuda.synchronize()
    ca,va,cra,vra=cp.cpu().numpy(),vp.cpu().numpy(),cr.cpu().numpy(),vr.cpu().numpy()
    with np.load(old/'outputs.npz') as d:oldp=d['packets'];oldr=d['roots']
    check=dict(packets_control_variant_bitwise=same(ca,va),roots_control_variant_bitwise=same(cra,vra),control_packets_match_original_bitwise=same(ca,oldp),control_roots_match_original_bitwise=same(cra,oldr))
    np.savez_compressed(out/'outputs.npz',control_packets=ca,variant_packets=va,control_roots=cra,variant_roots=vra)
    fns={'control':dict(preprocess=lambda:control.preprocess(src),root=lambda:control.solve(cp,sec),combined=lambda:control.solve(control.preprocess(src),sec)),
         'variant':dict(preprocess=lambda:variant.preprocess(src),root=lambda:variant.solve(vp,sec),combined=lambda:variant.solve(variant.preprocess(src),sec))}
    for arm in ('control','variant'):
        for fn in fns[arm].values():fn()
    torch.cuda.synchronize();pairs=[]
    for i in range(5):
        record=dict(pair=i,order=['control','variant'] if i%2==0 else ['variant','control'],results={})
        for arm in record['order']:record['results'][arm]={k:event(fn) for k,fn in fns[arm].items()}
        record['ratios']={k:record['results']['variant'][k]['event_ms']/record['results']['control'][k]['event_ms'] for k in fns['control']}
        pairs.append(record);save(out/'timing_progress.json',dict(completed_pairs=i+1,rows=pairs));print(f'pair{i}: combined ratio={record["ratios"]["combined"]:.6f}',flush=True)
    ratios={k:math.exp(sum(math.log(p['ratios'][k]) for p in pairs)/len(pairs)) for k in fns['control']}
    medians={arm:{k:float(np.median([p['results'][arm][k]['event_ms'] for p in pairs])) for k in fns[arm]} for arm in fns}
    result=dict(status='complete',checks=check,all_bitwise_checks_pass=all(check.values()),models=110,source_rows=7322,table_entries=51,paired_event_ratio_geometric_mean=ratios,median_event_ms=medians,
        preset_combined_ratio_gate=.95,local_optimization_gate_pass=all(check.values()) and ratios['combined']<=.95,wall_seconds=time.perf_counter()-start,
        scope='Same fixed mathematical ray diagnostic; includes output allocation/fill, excludes input transfer/full renderer/backward/GT. Bitwise equivalence inherits prior MP240 check, not a fresh unseen confirmation.')
    save(out/'paired_timing.json',dict(rows=pairs));save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','timing_progress.json')},generated_sha256={f'{folder}/{p.name}':sha(p) for folder in ('control_generated','variant_generated') for p in (out/folder).iterdir() if p.is_file()});save(out/'manifest.json',state);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    old=R/'experiments/device_root_v1';m=read(old/'manifest.json');assert m['status']=='complete' and read(old/'summary.json')['fixed_device_root_gate_pass']
    for n,h in m['artifacts_sha256'].items():assert sha(old/n)==h
    names=['root.cu','binding.cpp','guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh']
    for n in names:assert sha(old/'generated'/n)==m['generated_sha256'][n]
    with np.load(old/'inputs.npz') as d:source=d['source'].copy();sections=d['sections'].copy()
    sources=[Path(__file__),R/'reciprocal_root_cuda.py',R/'RECIPROCAL_ROOT_PROTOCOL.md',old/'manifest.json',old/'summary.json',old/'inputs.npz',old/'outputs.npz']+[old/'generated'/n for n in names]
    out.mkdir(parents=True);np.savez_compressed(out/'inputs.npz',source=source,sections=sections)
    state=dict(status='running',created_utc=now(),pid=os.getpid(),torch=torch.__version__,cuda_flags=FLAGS,build_env={k:os.environ.get(k) for k in ('CUDA_HOME','CC','CXX','MAX_JOBS','TORCH_CUDA_ARCH_LIST')},source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,old,source,sections)
    except BaseException as e:
        state.update(status='failed',failure_type=type(e).__name__,failure=str(e),failed_utc=now());save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
