"""All24 dynamic actual ID/position lists, image replay and full order checker."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,importlib.util,json,os,tempfile,time,traceback,zipfile
import numpy as np
import torch
from research.csr_capture_backend import load
from research.exact_perspective_backend import ROOT,RASTER
from research.exact_outlier_capture_rebuilt import native_args,parse
from research.benchmark_median_cuda import load_config
from research.benchmark_full_topology import digest

R=ROOT/'research';OLD=R/'experiments/full_topology_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def same(a,b):return bool(a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a.view(np.uint32),b.view(np.uint32)))
def event(fn):
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);start=time.perf_counter();a.record();result=fn();b.record();b.synchronize()
    return result,dict(event_ms=a.elapsed_time(b),wall_ms=1000*(time.perf_counter()-start))
def frames(native):return {name:native[i].detach().cpu().numpy() for name,i in [('rgb',1),('alpha',2),('normal',3),('depth_counters',4),('radii',5)]}
def pinned(meta):
    p=ROOT/meta['binary_relative_path'];assert sha(p)==meta['binary_sha256']
    spec=importlib.util.spec_from_file_location(p.stem,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def offsets_for(counts):
    cumulative=torch.cumsum(counts.reshape(-1).to(torch.int64),dim=0)
    offsets=torch.cat([torch.zeros(1,dtype=torch.int64,device=counts.device),cumulative])
    L=int(offsets[-1].item());records=torch.full((L,2),-1,dtype=torch.int32,device=counts.device)
    status=torch.zeros(counts.shape,dtype=torch.int32,device=counts.device)
    return offsets,records,status
def probe(out,args,counts,control_frames,module,change):
    modified=counts.clone();target=int(torch.nonzero(modified.reshape(-1)>0)[0].item());modified.reshape(-1)[target]+=change
    offsets,records,status=offsets_for(modified);native=module.rasterize_gaussians(*args,offsets,records,status);torch.cuda.synchronize()
    actual=frames(native);checks={k:same(control_frames[k],actual[k]) for k in control_frames};st=status.cpu().numpy()
    expected=3 if change==-1 else 2;flat=st.reshape(-1);target_only=bool(flat[target]==expected and np.count_nonzero(flat)==1)
    holes=int(torch.sum(records[:,0]==-1).item());holes_expected=0 if change==-1 else 1
    a=int(offsets[target].item());b=int(offsets[target+1].item());p=out/f'guard_{"under" if change==-1 else "over"}.npz'
    np.savez_compressed(p,write_status=st,offsets=offsets.cpu().numpy(),target_span_records=records[a:b].cpu().numpy())
    record=dict(change=change,target_flat_pixel=target,target_expected_status=expected,target_actual_status=int(flat[target]),
        nonzero_status_pixels=int(np.count_nonzero(flat)),unfilled_slots=holes,expected_unfilled_slots=holes_expected,
        original_output_bitwise_checks=checks,gate_pass=target_only and holes==holes_expected and all(checks.values()))
    save(p.with_suffix('.json'),record);return record

def run(out,state,prior):
    from scene import Scene,GaussianModel
    torch.set_num_threads(1);control=pinned(prior['backends']['instrumented']);start=time.perf_counter();module,meta=load(out,OLD)
    state.update(status='rendering',csr_backend=meta,build_seconds=time.perf_counter()-start,rows_completed=0);save(out/'manifest.json',state)
    rows=[];guards=[]
    for scan in [37,55,63]:
        ref=read(R/f'results/median_cpu_scan{scan}.json');names=list(dict.fromkeys(r['camera'] for r in ref['rows']));assert len(names)==8
        model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply'
        expected=prior['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected
        ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='csr-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
            gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False);P=int(gs.get_xyz.size(0));assert 0<P<=2**24
            cams={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()};bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            for vi,name in enumerate(names):
                camera=cams[name];W,H=camera.image_width,camera.image_height;oldrow=read(OLD/f'scan{scan}_{name}.json')
                save(out/'progress.json',dict(utc=now(),completed_views=len(rows),active_scan=scan,active_camera=name,phase='count_capture'))
                with torch.no_grad():
                    args=native_args(camera,gs,pipe,bg,ds.kernel_size)
                    if vi==0:assert {str(i):digest(args[i]) for i in range(11)}==prior['model_inputs'][str(scan)]['tensors']
                    assert {str(i):digest(args[i]) for i in [14,15,21]}==oldrow['camera_tensor_inputs']
                    n,count_time=event(lambda:control.rasterize_gaussians(*args));counts=n[4][3].to(torch.int64);control_frames=frames(n)
                    with np.load(OLD/f'scan{scan}_{name}.npz') as d:
                        assert np.array_equal(counts.cpu().numpy(),d['depth_accepted'])
                        assert same(control_frames['depth_counters'][:2],d['control_depth_slope'])
                    buffers,scan_time=event(lambda:offsets_for(counts));offsets,records,status=buffers;L=int(records.size(0))
                    torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
                    v,capture_time=event(lambda:module.rasterize_gaussians(*args,offsets,records,status))
                    memory=dict(allocated_before_capture_bytes=before,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                added_allocated_bytes=torch.cuda.max_memory_allocated()-before,records_bytes=records.numel()*records.element_size(),offsets_bytes=offsets.numel()*offsets.element_size(),status_bytes=status.numel()*status.element_size())
                    actual=frames(v);bitchecks={k:same(control_frames[k],actual[k]) for k in control_frames}
                    parsed,layout=parse(v,'affine',P,W,H)
                    checked,check_time=event(lambda:module.validate_csr(records,offsets,parsed['tile']['ranges'],parsed['binning']['point_list'],parsed['image']['n_contrib'],W,H,P))
                    status_ok=bool(torch.all(status==0).item());checker_ok=bool(torch.all(checked==0).item());filled_ok=bool(torch.all((records[:,0]>=0)&(records[:,0]<P)).item())
                    print(f'{scan}/{name}: records={L}, bitwise={all(bitchecks.values())}, status/check={status_ok}/{checker_ok}',flush=True)
                    if scan==37 and name=='0000':
                        guards=[probe(out,args,counts,control_frames,module,change) for change in [-1,1]]
                    save(out/'progress.json',dict(utc=now(),completed_views=len(rows),active_scan=scan,active_camera=name,phase='transfer_and_save'))
                    io_start=time.perf_counter();saved=dict(offsets=offsets.cpu().numpy(),records=records.cpu().numpy(),expected_counts=counts.cpu().numpy().astype(np.int32),
                        write_status=status.cpu().numpy(),checker_status=checked.cpu().numpy(),control_depth_counters=control_frames['depth_counters'],capture_depth_counters=actual['depth_counters'],
                        n_contrib=parsed['image']['n_contrib'].cpu().numpy(),tile_ranges=parsed['tile']['ranges'].cpu().numpy(),point_list=parsed['binning']['point_list'].cpu().numpy())
                    np.savez_compressed(out/f'scan{scan}_{name}.npz',**saved);io_wall=time.perf_counter()-io_start
                    del saved,parsed,layout,n,v,actual;torch.cuda.synchronize()
                    def capture():
                        records.fill_(-1);status.zero_();return module.rasterize_gaussians(*args,offsets,records,status)
                    fns={'count':lambda:control.rasterize_gaussians(*args),'capture':capture}
                    for fn in fns.values():
                        for _ in range(3):result=fn();del result
                    torch.cuda.synchronize();pairs=[]
                    for i in range(5):
                        pair=dict(pair=i,order=['count','capture'] if i%2==0 else ['capture','count'],results={})
                        for arm in pair['order']:result,t=event(fns[arm]);pair['results'][arm]=t;del result
                        pair['event_ratio']=pair['results']['capture']['event_ms']/pair['results']['count']['event_ms'];pairs.append(pair)
                    # Whole workflow includes count, scan/allocation, capture and checker.
                    def complete():
                        a=control.rasterize_gaussians(*args);o,r,s=offsets_for(a[4][3].to(torch.int64));b=module.rasterize_gaussians(*args,o,r,s)
                        q,_=parse(b,'affine',P,W,H);c=module.validate_csr(r,o,q['tile']['ranges'],q['binning']['point_list'],q['image']['n_contrib'],W,H,P)
                        return c,s
                    full_result,full_time=event(complete);assert bool(torch.all(full_result[0]==0).item()) and bool(torch.all(full_result[1]==0).item());del full_result
                row=dict(scan=scan,camera=name,width=W,height=H,primitives=P,pixels=W*H,records=L,original_output_bitwise_checks=bitchecks,
                    all_write_status_zero=status_ok,all_checker_status_zero=checker_ok,all_slots_filled_valid_ID=filled_ok,numerical_gate_pass=all(bitchecks.values()) and status_ok and checker_ok and filled_ok,
                    stage_time=dict(count=count_time,prefix_scan_allocation=scan_time,capture=capture_time,checker=check_time,whole_gpu_workflow=full_time,transfer_and_storage_wall_seconds=io_wall),
                    memory=memory,timing_pairs=pairs,event_median_ms={arm:float(np.median([p['results'][arm]['event_ms'] for p in pairs])) for arm in fns},
                    paired_capture_count_ratio_geometric_mean=float(np.exp(np.mean(np.log([p['event_ratio'] for p in pairs])))),
                    scope='Actual dynamic lists and source positions with all-pixel checker; no certified depth/backward/GT or training benefit.')
                save(out/f'scan{scan}_{name}.json',row);rows.append(row)
                save(out/'progress.json',dict(utc=now(),completed_views=len(rows),last_scan=scan,last_camera=name,phase='view_complete'))
                print(f'{len(rows)}/24 saved; numerical={row["numerical_gate_pass"]}, recordsBytes={memory["records_bytes"]}, complete event={full_time["event_ms"]:.3f}ms',flush=True)
                del args,counts,control_frames,buffers,offsets,records,status,checked,fns,complete,capture;torch.cuda.empty_cache()
            del scene,gs,cams;torch.cuda.empty_cache()
        assert sha(ply)==expected
    summary=dict(status='complete',views=len(rows),all_prespecified_24_views=len(rows)==24,pixels=sum(r['pixels'] for r in rows),records=sum(r['records'] for r in rows),
        numerical_gate_pass=len(rows)==24 and all(r['numerical_gate_pass'] for r in rows),guard_probe_pass=len(guards)==2 and all(g['gate_pass'] for g in guards),
        paired_capture_count_ratio_geometric_mean=float(np.exp(np.mean(np.log([r['paired_capture_count_ratio_geometric_mean'] for r in rows])))),
        rows=rows,guard_probes=guards,limits=['Full actual CSR lists only; no certified full-image solve, source backward, GT or training evidence.',
                'Paired capture times reuse CSR allocation; complete_workflow separately includes both forward passes and allocation/checker.',
                'Previously observed development scenes/views; no unseen population frequency inference.'])
    assert summary['records']==870178922
    save(out/'summary.json',summary)
    for name,h in state['source_sha256'].items():assert sha(ROOT/name)==h,name
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','guard_probes']},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve()
    if out.exists():raise FileExistsError(out)
    prior=read(OLD/'manifest.json');assert prior['status']=='complete';meta=prior['backends']['instrumented']
    for n,h in meta['generated_sha256'].items():assert sha(OLD/'instrumented_generated/src'/n)==h,n
    assert sha(ROOT/meta['binary_relative_path'])==meta['binary_sha256']
    sources=[Path(__file__),R/'csr_capture_backend.py',R/'CSR_CAPTURE_PROTOCOL.md',R/'benchmark_full_topology.py',R/'exact_outlier_capture_rebuilt.py',R/'exact_outlier_capture_backend.py',
             R/'exact_perspective_backend.py',R/'benchmark_median_cuda.py',OLD/'manifest.json',OLD/'summary.json']
    sources+=list((ROOT/'scene').glob('*.py'))+[ROOT/'arguments/__init__.py']
    sources += [OLD/'instrumented_generated/src'/n for n in meta['generated_sha256']]
    sources += [R/f'results/median_cpu_scan{scan}.json' for scan in [37,55,63]]
    out.mkdir(parents=True);state=dict(status='building',created_utc=now(),pid=os.getpid(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),control_backend=meta,
        source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources})
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state,prior)
    except BaseException as e:
        state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
