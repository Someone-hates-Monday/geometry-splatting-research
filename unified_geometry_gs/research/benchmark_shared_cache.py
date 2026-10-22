"""Exact fixed shared-cache parity, reuse timing and all24 real cache builds."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,tempfile,time,traceback,zipfile
import numpy as np
import torch
from research.shared_cache_cuda import load
from research.exact_perspective_backend import ROOT
from research.exact_outlier_capture_rebuilt import native_args
from research.benchmark_median_cuda import load_config
from research.benchmark_full_topology import digest
R=ROOT/'research'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()
def bits(a,b):return bool(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a.view(np.uint64),b.view(np.uint64)))
def event(fn):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);t=time.perf_counter();a.record();v=fn();b.record();b.synchronize()
    return v,dict(event_ms=a.elapsed_time(b),wall_ms=1000*(time.perf_counter()-t))
def tensor_sha(a):
    b=a.detach().cpu().contiguous().numpy();h=hashlib.sha256();h.update(str(b.dtype).encode());h.update(str(b.shape).encode());h.update(b.tobytes());return h.hexdigest()
def run(out,state,old):
    from scene import Scene,GaussianModel
    start=time.perf_counter();module=load(out,old)
    with np.load(old/'inputs.npz') as d:source=d['source'];sections=d['sections']
    src=torch.tensor(source,dtype=torch.float64,device='cuda');sec=torch.tensor(sections,dtype=torch.int32,device='cuda')
    ids=torch.arange(len(source),dtype=torch.int32,device='cuda');q=src[:,32:36].contiguous()
    original=module.preprocess(src);cache=module.build(src);decoded=module.decode(cache,ids,q)
    roots=module.solve(decoded,sec);torch.cuda.synchronize()
    a,b,c,r=[v.cpu().numpy() for v in [original,cache,decoded,roots]]
    with np.load(old/'outputs.npz') as d:prior_packets=d['variant_packets'];prior_roots=d['variant_roots']
    checks=dict(control_packets_prior=bits(a,prior_packets),cache_packets_control=bits(c,a),cache_roots_prior=bits(r,prior_roots))
    np.savez_compressed(out/'fixed_outputs.npz',cache=b,original_packets=a,decoded_packets=c,roots=r)
    repeated=src.repeat((8,1));ri=ids.repeat(8);rq=q.repeat((8,1));rp=module.preprocess(repeated);rd=module.decode(cache,ri,rq);torch.cuda.synchronize()
    checks['repeat8_packets']=bits(rp.cpu().numpy(),rd.cpu().numpy())
    def cached():return module.decode(module.build(src),ri,rq)
    fns={'original':lambda:module.preprocess(repeated),'cached':cached}
    for fn in fns.values():v=fn();del v
    pairs=[]
    for i in range(5):
        p=dict(pair=i,order=['original','cached'] if i%2==0 else ['cached','original'],results={})
        for key in p['order']:v,t=event(fns[key]);p['results'][key]=t;del v
        p['ratio']=p['results']['cached']['event_ms']/p['results']['original']['event_ms'];pairs.append(p)
    build_times=[];decode_times=[]
    for _ in range(5):
        v,t=event(lambda:module.build(src));build_times.append(t);del v
        v,t=event(lambda:module.decode(cache,ri,rq));decode_times.append(t);del v
    ratio=float(np.exp(np.mean(np.log([p['ratio'] for p in pairs]))))
    fixed=dict(source_rows=len(source),query_rows=8*len(source),checks=checks,all_bitwise_pass=all(checks.values()),finite_models=106,no_root_models=4,
        paired_ratio_geometric_mean=ratio,preset_ratio_gate=.95,local_reuse_gate_pass=all(checks.values()) and ratio<=.95,pairs=pairs,
        build_event_median_ms=float(np.median([p['event_ms'] for p in build_times])),decode_repeat8_event_median_ms=float(np.median([p['event_ms'] for p in decode_times])),
        control_repeat8_event_median_ms=float(np.median([p['results']['original']['event_ms'] for p in pairs])),cached_build_decode_event_median_ms=float(np.median([p['results']['cached']['event_ms'] for p in pairs])),
        build_times=build_times,decode_times=decode_times,scope='Fixed previously observed models, eight repeated queries; no new ray or full-render performance evidence.')
    save(out/'fixed_summary.json',fixed)
    if not all(checks.values()):raise AssertionError('Shared cache exact parity failed; saved fixed outputs retained')
    state.update(status='real_builds',fixed_local_reuse_gate_pass=fixed['local_reuse_gate_pass']);save(out/'manifest.json',state)
    del src,sec,ids,q,original,cache,decoded,roots,repeated,ri,rq,rp,rd;torch.cuda.empty_cache()
    topology=read(R/'experiments/full_topology_v1/manifest.json');real=[]
    for scan in [37,55,63]:
        reference=read(R/f'results/median_cpu_scan{scan}.json');names=list(dict.fromkeys(x['camera'] for x in reference['rows']));assert len(names)==8
        model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=topology['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected
        ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='shared-cache-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
            gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False);cams={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()}
            bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            for vi,name in enumerate(names):
                with torch.no_grad():
                    camera=cams[name];args=native_args(camera,gs,pipe,bg,ds.kernel_size)
                    if vi==0:assert {str(i):digest(args[i]) for i in range(11)}==topology['model_inputs'][str(scan)]['tensors']
                    oldrow=read(R/f'experiments/full_topology_v1/scan{scan}_{name}.json');assert {str(i):digest(args[i]) for i in [14,15,21]}==oldrow['camera_tensor_inputs']
                    P=int(args[1].size(0));src=torch.zeros((P,36),dtype=torch.float64,device='cuda')
                    src[:,:3]=args[1].double();src[:,3:6]=args[4].double();src[:,6:10]=args[5].double();src[:,10]=1
                    src[:,20:29]=args[14][:3,:3].T.reshape(1,9).double();src[:,29:32]=args[14][3,:3].reshape(1,3).double();src[:,34]=1
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
                    result,t=event(lambda:module.build(src));memory=dict(allocated_before_bytes=before,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        added_allocated_bytes=torch.cuda.max_memory_allocated()-before,source_rows_bytes=src.numel()*src.element_size(),cache_bytes=result.numel()*result.element_size(),cache_row_bytes=result.stride(0)*result.element_size())
                    data=result.cpu().numpy();statuses={str(mode):{str(int(code)):int(np.sum(data[:,mode]==code)) for code in np.unique(data[:,mode])} for mode in [0,1]}
                    badrows=int(np.sum(~np.isfinite(data).all(axis=1)));constructible=bool(np.all(data[:,:2]==0) and badrows==0)
                    hashvalue=tensor_sha(result);eta_stats={str(mode):dict(min=float(data[:,3+mode].min()),p50=float(np.median(data[:,3+mode])),max=float(data[:,3+mode].max())) for mode in [0,1]} if badrows==0 else None
                    del result;times=[t]
                    for _ in range(4):v,tt=event(lambda:module.build(src));times.append(tt);del v
                    row=dict(scan=scan,camera=name,primitives=P,statuses=statuses,nonfinite_cache_rows=badrows,all_cache_constructible=constructible,cache_shape=list(data.shape),cache_content_sha256=hashvalue,
                        memory=memory,event_ms=[x['event_ms'] for x in times],wall_ms=[x['wall_ms'] for x in times],median_event_ms=float(np.median([x['event_ms'] for x in times])),eta_stats=eta_stats,
                        scope='Complete primitive/camera cache builds only; query/AMP placeholders ignored, no full CSR ray decode/root/backward/GT.')
                    save(out/f'scan{scan}_{name}.json',row);real.append(row);save(out/'progress.json',dict(utc=now(),completed_views=len(real),last_scan=scan,last_camera=name))
                    print(f'{len(real)}/24 cache scan{scan}/{name}: P={P}, bytes={memory["cache_bytes"]}, build={row["median_event_ms"]:.3f}ms, constructible={constructible}',flush=True)
                    del src,data,args;torch.cuda.empty_cache()
            del scene,gs,cams;torch.cuda.empty_cache()
        assert sha(ply)==expected
    summary=dict(status='complete',fixed=fixed,real_views=len(real),all24_real_builds=len(real)==24,all_real_cache_constructible=all(r['all_cache_constructible'] for r in real),
        cache_row_bytes=224,rows=real,wall_seconds=time.perf_counter()-start,limits=['Diagnostic224B cache, not the proposed56B production layout.',
            'All real primitive caches built; no all-record ray decode, full root/certified images or source VJP/GT/training proof.'])
    save(out/'summary.json',summary)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h,n
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']},generated_sha256={p.name:sha(p) for p in (out/'generated').iterdir() if p.is_file()});save(out/'manifest.json',state)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','fixed']},indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve()
    if out.exists():raise FileExistsError(out)
    old=R/'experiments/reciprocal_root_v2';m=read(old/'manifest.json');assert m['status']=='complete'
    for n in ['guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh']:assert sha(old/'variant_generated'/n)==m['generated_sha256']['variant_generated/'+n]
    for n in ['inputs.npz','outputs.npz']:assert sha(old/n)==m['artifacts_sha256'][n]
    sources=[Path(__file__),R/'shared_cache_cuda.py',R/'SHARED_CACHE_PROTOCOL.md',R/'guarded_interval_cuda.py',R/'exact_outlier_capture_rebuilt.py',R/'exact_outlier_capture_backend.py',R/'exact_perspective_backend.py',R/'benchmark_median_cuda.py',R/'benchmark_full_topology.py',old/'manifest.json',old/'inputs.npz',old/'outputs.npz']
    sources += [old/'variant_generated'/n for n in ['guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh']]+list((ROOT/'scene').glob('*.py'))
    sources += [R/f'results/median_cpu_scan{scan}.json' for scan in [37,55,63]]+[R/'experiments/full_topology_v1/manifest.json']
    out.mkdir(parents=True);state=dict(status='building',created_utc=now(),pid=os.getpid(),torch=torch.__version__,source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,old)
    except BaseException as e:
        state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
