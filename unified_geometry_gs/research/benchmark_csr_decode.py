"""All actual CSR associations streamed through the frozen shared decoder."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,tempfile,time,traceback,zipfile
import numpy as np
import torch
from research.csr_decode_cuda import load_query,load_stream
from research.benchmark_csr_capture import pinned
from research.exact_perspective_backend import ROOT,RASTER
from research.exact_outlier_capture_rebuilt import native_args,parse
from research.benchmark_median_cuda import load_config
from research.benchmark_full_topology import digest
R=ROOT/'research';TOPO=R/'experiments/full_topology_v1';CSR=R/'experiments/csr_capture_v1';CACHE=R/'experiments/shared_cache_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def bits(a,b):return bool(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint32 if a.dtype.itemsize==4 else np.uint64),b.view(np.uint32 if b.dtype.itemsize==4 else np.uint64)))
def event(fn):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);t=time.perf_counter();a.record();v=fn();b.record();b.synchronize();return v,dict(event_ms=a.elapsed_time(b),wall_ms=1000*(time.perf_counter()-t))
def src_rows(args):
    P=int(args[1].size(0));src=torch.zeros((P,36),dtype=torch.float64,device='cuda');src[:,:3]=args[1].double();src[:,3:6]=args[4].double();src[:,6:10]=args[5].double();src[:,10]=1
    src[:,20:29]=args[14][:3,:3].T.reshape(1,9).double();src[:,29:32]=args[14][3,:3].reshape(1,3).double();src[:,34]=1;return src
def field_frames(n):return {name:n[i].detach().cpu().numpy() for name,i in [('rgb',1),('alpha',2),('normal',3),('depth',4),('radii',5)]}
def run(out,state):
    from scene import Scene,GaussianModel
    torch.set_num_threads(1);topology=read(TOPO/'manifest.json');csr_manifest=read(CSR/'manifest.json');cache_manifest=read(CACHE/'manifest.json')
    control=pinned(topology['backends']['instrumented'])
    bfile=next(n for n in cache_manifest['generated_sha256'] if n.endswith('.so'));cache_meta=dict(binary_relative_path=(CACHE/'generated'/bfile).relative_to(ROOT).as_posix(),binary_sha256=cache_manifest['generated_sha256'][bfile]);geometry=pinned(cache_meta)
    capture,capture_meta=load_query(out,TOPO);stream,stream_meta=load_stream(out,CACHE)
    state.update(status='decoding',capture_backend=capture_meta,stream_backend=stream_meta,cache_backend=cache_meta);save(out/'manifest.json',state)
    old_cases=read(R/'experiments/exact_outlier_capture_v3/summary.json')['rows'];views=[];total=0
    for scan in [37,55,63]:
        reference=read(R/f'results/median_cpu_scan{scan}.json');names=list(dict.fromkeys(x['camera'] for x in reference['rows']));model=ROOT/f'output_compare/ggs_pipeline/scan{scan}'
        ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=topology['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected
        ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='csr-decode-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
            gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False);cams={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()};bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            for vi,name in enumerate(names):
                view_start=time.perf_counter();save(out/'progress.json',dict(utc=now(),completed_views=len(views),active_scan=scan,active_camera=name,phase='capture'))
                with torch.no_grad():
                    cam=cams[name];args=native_args(cam,gs,pipe,bg,ds.kernel_size);P=int(args[1].size(0));W,H=cam.image_width,cam.image_height
                    if vi==0:assert {str(i):digest(args[i]) for i in range(11)}==topology['model_inputs'][str(scan)]['tensors']
                    oldrow=read(TOPO/f'scan{scan}_{name}.json');assert {str(i):digest(args[i]) for i in [14,15,21]}==oldrow['camera_tensor_inputs']
                    n=control.rasterize_gaussians(*args);v,capture_time=event(lambda:capture.rasterize_gaussians(*args));oldframes=field_frames(n);frames=field_frames(v)
                    same_image={k:bits(oldframes[k],frames[k][:6] if k=='depth' else frames[k]) for k in oldframes};assert all(same_image.values())
                    parsed,_=parse(v,'affine',P,W,H);amp=parsed['geom']['conic'][:,3].contiguous();pixelq=v[4][6:8].permute(1,2,0).reshape(-1,2).contiguous()
                    query_cpu=pixelq.cpu().numpy();amp_cpu=amp.cpu().numpy();assert np.isfinite(query_cpu).all() and np.isfinite(amp_cpu).all() and (amp_cpu>=0).all()
                    csrfile=CSR/f'scan{scan}_{name}.npz';assert sha(csrfile)==csr_manifest['artifacts_sha256'][csrfile.name]
                    with np.load(csrfile) as d:records_cpu=d['records'];offsets_cpu=d['offsets'];counts_cpu=d['expected_counts'];assert np.array_equal(counts_cpu,frames['depth'][3])
                    records=torch.tensor(records_cpu,dtype=torch.int32,device='cuda');offsets=torch.tensor(offsets_cpu,dtype=torch.int64,device='cuda');L=len(records_cpu)
                    src=src_rows(args);cache,build_time=event(lambda:geometry.build(src));assert bool(torch.all(cache[:,:2]==0).item()) and bool(torch.isfinite(cache).all().item())
                    indices=set(np.linspace(0,L-1,48,dtype=np.int64).tolist());special=[]
                    for case in old_cases:
                        if case['scan']==scan and case['camera']==name:
                            x,y=case['pixel'];pixel=y*W+x;indices.update(range(int(offsets_cpu[pixel]),int(offsets_cpu[pixel+1])))
                            qsaved=case['copied_initialization']['qray_xy'];special.append(dict(case=case['case'],pixel=[x,y],actual_query=query_cpu[pixel].tolist(),old_query=qsaved,query_bits_match=bits(query_cpu[pixel],np.array(qsaved,np.float32))))
                    chosen=np.array(sorted(indices),np.int64);probe_packets=[];probe_sources=[];probe_indices=[]
                    # Match sampled direct preprocessing independently of stream indexing.
                    pixel_for_probe=np.searchsorted(offsets_cpu,chosen,side='right')-1;point_for_probe=records_cpu[chosen,0]
                    prosrc=src[torch.tensor(point_for_probe,device='cuda',dtype=torch.long)].clone();prosrc[:,32:34]=torch.tensor(query_cpu[pixel_for_probe],device='cuda',dtype=torch.float64);prosrc[:,34]=1;prosrc[:,35]=torch.tensor(amp_cpu[point_for_probe],device='cuda',dtype=torch.float64)
                    direct=geometry.preprocess(prosrc).cpu().numpy();probe_map={int(i):j for j,i in enumerate(chosen)};saved_probe=np.empty_like(direct)
                    batches=[];counts=[{},{}];nonfinite=[0,0];candidate_zero=[0,0];event_total=0.;cpu_total=0.;max_added=0
                    for start in range(0,L,65536):
                        count=min(65536,L-start);torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated();packet,t=event(lambda:stream.stream(cache,records,offsets,pixelq,amp,start,count))
                        max_added=max(max_added,torch.cuda.max_memory_allocated()-before);cpu_start=time.perf_counter();data=packet.cpu().numpy();hashvalue=hashlib.sha256(data.tobytes()).hexdigest()
                        for mode in [0,1]:
                            statuses=data[:,mode,0];u,c=np.unique(statuses,return_counts=True)
                            for code,nc in zip(u,c):counts[mode][str(int(code))]=counts[mode].get(str(int(code)),0)+int(nc)
                            valid=statuses==0;nonfinite[mode]+=int(np.sum(valid&~np.isfinite(data[:,mode]).all(axis=1)))
                            a=data[:,mode,7]*np.exp(-data[:,mode,10]/2);candidate_zero[mode]+=int(np.sum(valid&(a==0)))
                        hit=chosen[(chosen>=start)&(chosen<start+count)]
                        for index in hit:saved_probe[probe_map[int(index)]]=data[int(index)-start]
                        cpu_wall=time.perf_counter()-cpu_start;cpu_total+=cpu_wall;event_total+=t['event_ms']
                        batches.append(dict(start=start,count=count,output_sha256=hashvalue,event_ms=t['event_ms'],wall_ms=t['wall_ms'],transfer_hash_status_wall_seconds=cpu_wall))
                        if len(batches)%64==0 or start+count==L:
                            save(out/'progress.json',dict(utc=now(),completed_views=len(views),active_scan=scan,active_camera=name,processed_in_view=start+count,view_records=L,phase='streaming'))
                        del packet,data
                    probe_bitwise=bits(saved_probe,direct);assert probe_bitwise
                    np.savez_compressed(out/f'scan{scan}_{name}_probes.npz',source=prosrc.cpu().numpy(),packets=saved_probe,direct_packets=direct,record_indices=chosen,pixels=pixel_for_probe,primitive_ids=point_for_probe)
                    np.savez_compressed(out/f'scan{scan}_{name}_query_amp.npz',query_xy=query_cpu,amplitude=amp_cpu,counters=frames['depth'][:6])
                    high_safe=counts[1].get('0',0)==L and nonfinite[1]==0
                    row=dict(scan=scan,camera=name,width=W,height=H,primitives=P,records=L,processed_records=sum(b['count'] for b in batches),original_image_bitwise=same_image,
                        counts_match_saved_CSR=True,status_counts=counts,status0_nonfinite=nonfinite,nominal_alpha_underflow_zero=candidate_zero,all_high_path_valid=high_safe,
                        probe_count=len(chosen),probe_cached_direct_bitwise=probe_bitwise,old_anomaly_queries=special,batches=batches,
                        decode_event_total_ms=event_total,transfer_hash_status_wall_seconds=cpu_total,view_wall_seconds=time.perf_counter()-view_start,
                        capture_time=capture_time,cache_build_time=build_time,max_batch_added_allocated_bytes=max_added,batch_output_bytes=65536*2*26*8,
                        scope='All actual CSR decode states, fixed probe parity; no root/certificate/backward/GT evidence.')
                    save(out/f'scan{scan}_{name}.json',row);views.append(row);total+=L
                    save(out/'progress.json',dict(utc=now(),completed_views=len(views),processed_total=total,last_scan=scan,last_camera=name,phase='view_complete'))
                    print(f'{len(views)}/24 decode scan{scan}/{name}: {L} records, states={counts}, nonfinite={nonfinite}, event={event_total:.1f}ms',flush=True)
                    del args,n,v,parsed,src,cache,records,offsets,pixelq,amp,records_cpu,offsets_cpu,query_cpu,amp_cpu,prosrc,direct,saved_probe;torch.cuda.empty_cache()
            del scene,gs,cams;torch.cuda.empty_cache()
        assert sha(ply)==expected
    result=dict(status='complete',views=len(views),records=total,all24_views=len(views)==24,complete_record_count_pass=total==870178922,
        all_high_path_valid=all(r['all_high_path_valid'] for r in views),all_fixed_probes_bitwise=all(r['probe_cached_direct_bitwise'] for r in views),rows=views,
        independent_MP_world_probe_audit='pending separate verifier',scope='Complete actual record decode/statistics only; root/backward/GT/training not verified.')
    save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h,n
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',state)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve()
    if out.exists():raise FileExistsError(out)
    sources=[Path(__file__),R/'csr_decode_cuda.py',R/'CSR_DECODE_PROTOCOL.md',R/'benchmark_csr_capture.py',R/'exact_outlier_capture_rebuilt.py',R/'exact_outlier_capture_backend.py',R/'exact_perspective_backend.py',R/'benchmark_median_cuda.py',R/'benchmark_full_topology.py',TOPO/'manifest.json',CSR/'manifest.json',CSR/'summary.json',CACHE/'manifest.json',R/'experiments/exact_outlier_capture_v3/summary.json']
    sources += list((ROOT/'scene').glob('*.py'))+[CACHE/'generated'/n for n in ['guarded_interval.cuh','guarded_preprocess.cuh','shared_cache.cuh']]
    sources += list((TOPO/'instrumented_generated/src').rglob('*.cu'))+list((TOPO/'instrumented_generated/src').rglob('*.cpp'))+list((TOPO/'instrumented_generated/src').rglob('*.h'))
    out.mkdir(parents=True);state=dict(status='building',created_utc=now(),pid=os.getpid(),torch=torch.__version__,source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state)
    except BaseException as e:
        state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
