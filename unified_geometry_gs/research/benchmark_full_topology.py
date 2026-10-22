"""All24 native affine images: actual acceptance counts and paired overhead."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, tempfile, time, traceback, zipfile
import numpy as np
import torch
from research.full_topology_backend import load
from research.exact_perspective_backend import ROOT, RASTER
from research.exact_outlier_capture_rebuilt import native_args, parse
from research.benchmark_median_cuda import load_config

R=ROOT/'research'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def bits(a,b):return bool(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint32),b.view(np.uint32)))
def digest(t):
    a=t.detach().contiguous().cpu().numpy();h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=h.hexdigest())
def distribution(counts,mask):
    a=counts[mask];n=int(a.size)
    if n==0:return dict(pixels=0)
    hist=np.bincount(a)
    return dict(pixels=n,min=int(a.min()),p50=float(np.quantile(a,.5)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)),max=int(a.max()),
                total_accepted=int(a.sum(dtype=np.int64)),histogram={str(i):int(v) for i,v in enumerate(hist) if v},
                capacity_exceed={str(k):dict(pixels=int((a>k).sum()),fraction=float((a>k).mean())) for k in [192,256,512]})
def event(fn):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);start=time.perf_counter();a.record();result=fn();b.record();b.synchronize()
    del result
    return dict(event_ms=a.elapsed_time(b),wall_ms=1000*(time.perf_counter()-start))

def run(out,state,smoke):
    from scene import Scene, GaussianModel
    torch.set_num_threads(1);build_start=time.perf_counter();modules={};metadata={}
    for key,instrumented in [('control',False),('instrumented',True)]:
        print('Building '+key,flush=True);modules[key],metadata[key]=load(out,instrumented)
    state.update(status='rendering',backends=metadata,build_seconds=time.perf_counter()-build_start,model_inputs={})
    save(out/'manifest.json',state);rows=[];all_initial=[];all_valid=[]
    for scan in [37,55,63]:
        refpath=ROOT/f'research/results/median_cpu_scan{scan}.json';ref=read(refpath)
        model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply'
        expected=next(v for k,v in ref['source_sha256'].items() if k.endswith('point_cloud.ply'));assert sha(ply)==expected
        names=list(dict.fromkeys(row['camera'] for row in ref['rows']));assert len(names)==8
        if smoke:names=names[:1]
        ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='topology-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
            gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
            P=int(gs.get_xyz.size(0));assert 0<P<=2**24
            cameras={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()}
            bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            for view_index,name in enumerate(names):
                camera=cameras[name];W,H=camera.image_width,camera.image_height
                with torch.no_grad():
                    args=native_args(camera,gs,pipe,bg,ds.kernel_size)
                    if view_index==0:
                        state['model_inputs'][str(scan)]=dict(ply_relative_path=ply.relative_to(ROOT).as_posix(),ply_sha256=expected,
                            primitives=P,tensors={str(i):digest(args[i]) for i in range(11)})
                        save(out/'manifest.json',state)
                    camera_inputs={str(i):digest(args[i]) for i in [14,15,21]}
                    camera_scalars={str(i):args[i] for i in [11,12,13,16,17,18,19,20,22,23,24]}
                    native={};memory={}
                    for arm,module in modules.items():
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
                        native[arm]=module.rasterize_gaussians(*args);torch.cuda.synchronize()
                        memory[arm]=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            allocated_before_bytes=before,peak_added_allocated_bytes=torch.cuda.max_memory_allocated()-before)
                    c,v=native['control'],native['instrumented'];assert c[4].shape==(2,H,W) and v[4].shape==(6,H,W)
                    framec={key:c[index].detach().cpu().numpy() for key,index in [('rgb',1),('alpha',2),('normal',3),('depth_slope',4),('radii',5)]}
                    framev={key:(v[index][:2] if index==4 else v[index]).detach().cpu().numpy() for key,index in [('rgb',1),('alpha',2),('normal',3),('depth_slope',4),('radii',5)]}
                    identical={key:bits(framec[key],framev[key]) for key in framec}
                    finite=all(np.isfinite(x).all() for x in [*framec.values(),*framev.values()])
                    counters=v[4][2:].detach().cpu().numpy();counter_valid=bool(np.isfinite(counters).all() and (counters>=0).all() and (counters==np.floor(counters)).all())
                    rgb_count=counters[0].astype(np.int32);depth_count=counters[1].astype(np.int32);nonball=counters[2].astype(np.int32);initial=counters[3]==1
                    parsed,layout=parse(c,'affine',P,W,H);ncontrib=parsed['image']['n_contrib'].detach().cpu().numpy()
                    valid=framec['depth_slope'][0]>0
                    count_bounds=bool((rgb_count<=P).all() and (rgb_count<=ncontrib).all() and (depth_count<=P).all() and (nonball<=depth_count).all()
                                      and np.isin(counters[3],[0,1]).all() and (depth_count[~initial]==0).all())
                    mismatch=initial&(rgb_count!=depth_count)
                    numerical=finite and all(identical.values()) and counter_valid and count_bounds
                    # Preserve full fields/positions; original invalids are not repaired.
                    np.savez_compressed(out/f'scan{scan}_{name}.npz',control_depth_slope=framec['depth_slope'],instrumented_depth_slope=framev['depth_slope'],
                        rgb_accepted=rgb_count,depth_accepted=depth_count,nonball_accepted=nonball,initial_eligible=initial,final_valid=valid,n_contrib=ncontrib,
                        rgb_depth_count_mismatch=mismatch)
                    del parsed,layout,c,v,native;torch.cuda.synchronize()
                    fns={arm:(lambda module=module:module.rasterize_gaussians(*args)) for arm,module in modules.items()}
                    for fn in fns.values():
                        for _ in range(3):fn()
                    torch.cuda.synchronize();pairs=[]
                    for i in range(5):
                        record=dict(pair=i,order=['control','instrumented'] if i%2==0 else ['instrumented','control'],results={})
                        for arm in record['order']:record['results'][arm]=event(fns[arm])
                        record['event_ratio']=record['results']['instrumented']['event_ms']/record['results']['control']['event_ms'];pairs.append(record)
                row=dict(scan=scan,camera=name,width=W,height=H,primitives=P,pixels=W*H,original_output_bitwise_checks=identical,all_original_outputs_finite=bool(finite),
                    counter_nonnegative_integer_pass=counter_valid,count_bounds_pass=count_bounds,numerical_gate_pass=bool(numerical),
                    initial_eligible_pixels=int(initial.sum()),final_valid_pixels=int(valid.sum()),rgb_depth_count_mismatches=int(mismatch.sum()),
                    initial_depth_counts=distribution(depth_count,initial),valid_depth_counts=distribution(depth_count,valid),all_rgb_counts=distribution(rgb_count,np.ones((H,W),bool)),
                    initial_nonball_pixels=int((initial&(nonball>0)).sum()),valid_nonball_pixels=int((valid&(nonball>0)).sum()),
                    memory=memory,timing_pairs=pairs,event_median_ms={arm:float(np.median([p['results'][arm]['event_ms'] for p in pairs])) for arm in modules},
                    paired_event_ratio_geometric_mean=float(np.exp(np.mean(np.log([p['event_ratio'] for p in pairs])))),camera_tensor_inputs=camera_inputs,camera_scalar_inputs=camera_scalars,
                    scope='Actual full native affine acceptance context. Counts only; not certified depth, GT/backward or training performance.')
                save(out/f'scan{scan}_{name}.json',row);rows.append(row);all_initial.append(depth_count[initial]);all_valid.append(depth_count[valid])
                save(out/'progress.json',dict(utc=now(),completed_views=len(rows),last_scan=scan,last_camera=name))
                print(f'{len(rows)}/24 scan{scan}/{name}: bitwise={all(identical.values())}, initial max={row["initial_depth_counts"].get("max")}, valid >192={row["valid_depth_counts"].get("capacity_exceed",{}).get("192")}',flush=True)
                del args,fns,framec,framev,camera_inputs,counters;torch.cuda.empty_cache()
            del scene,gs,cameras;torch.cuda.empty_cache()
        assert sha(ply)==expected
    ai=np.concatenate(all_initial);av=np.concatenate(all_valid)
    summary=dict(status='complete',phase='smoke3' if smoke else 'full24',views=len(rows),all_prespecified_24_views=not smoke and len(rows)==24,
        all_original_outputs_bitwise=all(all(r['original_output_bitwise_checks'].values()) for r in rows),numerical_gate_pass=all(r['numerical_gate_pass'] for r in rows),
        pixels=sum(r['pixels'] for r in rows),rgb_depth_count_mismatches=sum(r['rgb_depth_count_mismatches'] for r in rows),
        initial_depth_counts=distribution(ai,np.ones(ai.shape,bool)),valid_depth_counts=distribution(av,np.ones(av.shape,bool)),
        capacity192_covers_all_initial=bool((ai<=192).all()),capacity192_covers_all_final_valid=bool((av<=192).all()),
        paired_count_instrumentation_ratio_geometric_mean=float(np.exp(np.mean(np.log([r['paired_event_ratio_geometric_mean'] for r in rows])))),
        rows=rows,limits=['Native forward only; postprocessing appearance, certified cache/solver, unresolved policy, backward and GT are not tested.',
                          'All previously observed development views; counts do not establish independent dataset prevalence.',
                          'Event/wall and memory measure counting overhead, not the short-certificate or full-training method.'])
    save(out/'summary.json',summary)
    for name,digest_ in state['source_sha256'].items():assert sha(ROOT/name)==digest_,name
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','initial_depth_counts','valid_depth_counts')},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke3',action='store_true');args=p.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    sources=[Path(__file__),R/'full_topology_backend.py',R/'FULL_TOPOLOGY_PROTOCOL.md',R/'exact_perspective_backend.py',R/'exact_outlier_capture_rebuilt.py',R/'exact_outlier_capture_backend.py',R/'benchmark_median_cuda.py']
    sources+=list((ROOT/'scene').glob('*.py'))+[ROOT/'arguments/__init__.py']
    sources += [p for p in RASTER.rglob('*') if p.is_file() and p.suffix in ('.cu','.cpp','.h','.hpp','.inl')]
    sources += [R/f'results/median_cpu_scan{scan}.json' for scan in [37,55,63]]
    out.mkdir(parents=True);state=dict(status='building',created_utc=now(),pid=os.getpid(),phase='smoke3' if args.smoke3 else 'full24',torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for source in sources:z.write(source,source.relative_to(ROOT).as_posix())
    try:run(out,state,args.smoke3)
    except BaseException as e:
        state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
