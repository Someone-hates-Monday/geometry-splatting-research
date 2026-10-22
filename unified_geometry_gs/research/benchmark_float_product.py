"""Predeclared full CSR F32 product certificate coverage and timing."""
from pathlib import Path
from fractions import Fraction
import argparse,hashlib,json,os,tempfile,time,traceback,zipfile
import numpy as np
import torch
import mpmath as mp
from research.float_product_cuda import load
from research.benchmark_csr_decode import src_rows,read,sha,save,now,event
from research.benchmark_csr_capture import pinned
from research.exact_perspective_backend import ROOT
from research.exact_outlier_capture_rebuilt import native_args
from research.benchmark_median_cuda import load_config
from research.benchmark_full_topology import digest
R=ROOT/'research';CACHE=R/'experiments/shared_cache_v1';CSR=R/'experiments/csr_capture_v1';DECODE=R/'experiments/csr_decode_v1';TOPO=R/'experiments/full_topology_v1'

def scalar_check(module,out):
    mp.mp.dps=100
    lnlo=Fraction(float.fromhex('0x1.62e42ep-1'));lnhi=Fraction(float.fromhex('0x1.62e430p-1'))
    term=Fraction(1,3);s=Fraction(0)
    for j in range(50):s+=2*term/(2*j+1);term/=9
    rem=2*term/(101*(1-Fraction(1,9)))
    assert lnlo<s<s+rem<lnhi and 115*lnhi<80 and Fraction(float.fromhex('0x1.0624dd2f1a9fbp-10'))<Fraction(1,1000)
    rng=np.random.default_rng(13017);xs=np.concatenate([np.linspace(-80,0,2049,dtype=np.float32),rng.uniform(-80,0,2048).astype(np.float32),np.array([-1e30,-1000,-100,-80,np.nextafter(np.float32(-80),np.float32(-np.inf)),-0.,-np.inf],np.float32)])
    ys=module.scalars(torch.tensor(xs,device='cuda')).cpu().numpy();fail=[]
    for i,(x,y) in enumerate(zip(xs,ys)):
        value=mp.exp(mp.mpf(float(x)))
        if not mp.mpf(float(y[0]))<=value<=mp.mpf(float(y[1])):fail.append(i)
    np.savez_compressed(out/'scalars.npz',x=xs,y=ys)
    result=dict(count=len(xs),failures=fail,rational_ln2_pass=True,tail_bound_pass=True,tolerance_bound_pass=True)
    save(out/'scalar_check.json',result);assert not fail,result

def run(out,state):
    from scene import Scene,GaussianModel
    torch.set_num_threads(1);top=read(TOPO/'manifest.json');cm=read(CACHE/'manifest.json');dm=read(DECODE/'manifest.json');csrm=read(CSR/'manifest.json')
    bfile=next(n for n in cm['generated_sha256'] if n.endswith('.so'));meta=dict(binary_relative_path=(CACHE/'generated'/bfile).relative_to(ROOT).as_posix(),binary_sha256=cm['generated_sha256'][bfile]);geometry=pinned(meta)
    module,backend=load(out,CACHE);state.update(status='scalar_checks',backend=backend,cache_backend=meta);save(out/'manifest.json',state);scalar_check(module,out)
    oldcases=read(R/'experiments/exact_outlier_capture_v3/summary.json')['rows'];rows=[]
    for scan in [37,55,63]:
        names=list(dict.fromkeys(x['camera'] for x in read(R/f'results/median_cpu_scan{scan}.json')['rows']));model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=top['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected;ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='float-product-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch);gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False);cams={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()};bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            for vi,name in enumerate(names):
                start=time.perf_counter();save(out/'progress.json',dict(utc=now(),completed_views=len(rows),active_scan=scan,active_camera=name,phase='preparing'))
                with torch.no_grad():
                    cam=cams[name];args=native_args(cam,gs,pipe,bg,ds.kernel_size);W,H=cam.image_width,cam.image_height
                    if vi==0:assert {str(i):digest(args[i]) for i in range(11)}==top['model_inputs'][str(scan)]['tensors']
                    assert {str(i):digest(args[i]) for i in [14,15,21]}==read(TOPO/f'scan{scan}_{name}.json')['camera_tensor_inputs']
                    datafile=DECODE/f'scan{scan}_{name}_query_amp.npz';csrfile=CSR/f'scan{scan}_{name}.npz';assert sha(datafile)==dm['artifacts_sha256'][datafile.name];assert sha(csrfile)==csrm['artifacts_sha256'][csrfile.name]
                    with np.load(datafile) as d:query=d['query_xy'];amp=d['amplitude'];counters=d['counters']
                    with np.load(csrfile) as d:records=d['records'];offsets=d['offsets'];assert np.array_equal(d['expected_counts'],counters[3])
                    assert len(query)==W*H;z=counters[0].reshape(-1);eligible=counters[5].reshape(-1);src=src_rows(args);cache,build_time=event(lambda:geometry.build(src));assert bool(torch.all(cache[:,:2]==0).item())
                    inputs=[cache,torch.tensor(records,dtype=torch.int32,device='cuda'),torch.tensor(offsets,dtype=torch.int64,device='cuda'),torch.tensor(query,device='cuda'),torch.tensor(amp,device='cuda'),torch.tensor(z,device='cuda'),torch.tensor(eligible,device='cuda')]
                    save(out/'progress.json',dict(utc=now(),completed_views=len(rows),active_scan=scan,active_camera=name,phase='certificate'))
                    initial,t=event(lambda:module.run(*inputs));data=initial.cpu().numpy();timings=[]
                    for repeat in range(3):
                        value,timing=event(lambda:module.run(*inputs));assert np.array_equal(value.cpu().numpy().view(np.uint64),data.view(np.uint64));timings.append(timing);del value
                    codes,counts=np.unique(data[:,0],return_counts=True);stats={str(int(k)):int(v) for k,v in zip(codes,counts)}
                    indices=np.flatnonzero(eligible==1);chosen=set(indices[np.linspace(0,len(indices)-1,min(64,len(indices)),dtype=np.int64)].tolist());anomalies=[]
                    for case in oldcases:
                        if case['scan']==scan and case['camera']==name:
                            x,y=case['pixel'];p=y*W+x;chosen.add(p);anomalies.append(dict(case=case['case'],pixel=p,xy=[x,y]))
                    chosen=np.array(sorted(chosen),np.int64);recidx=np.concatenate([np.arange(offsets[p],offsets[p+1],dtype=np.int64) for p in chosen]);ids=records[recidx,0];pixel_for_record=np.repeat(chosen,np.diff(offsets)[chosen]);prosrc=src[torch.tensor(ids,dtype=torch.long,device='cuda')].clone();prosrc[:,32:34]=torch.tensor(query[pixel_for_record],dtype=torch.float64,device='cuda');prosrc[:,34]=1;prosrc[:,35]=torch.tensor(amp[ids],dtype=torch.float64,device='cuda');probeout=module.probes(prosrc,cache,torch.tensor(ids,dtype=torch.int32,device='cuda')).cpu().numpy()
                    np.savez_compressed(out/f'scan{scan}_{name}_output.npz',output=data)
                    np.savez_compressed(out/f'scan{scan}_{name}_probes.npz',pixels=chosen,source=prosrc.cpu().numpy(),packets=probeout,sections=np.concatenate([[0],np.cumsum(np.diff(offsets)[chosen])]),records=recidx,ids=ids,query=query[chosen],output=data[chosen])
                    row=dict(scan=scan,camera=name,width=W,height=H,records=len(records),pixels=W*H,initial_eligible=int(np.sum(eligible==1)),original_valid=int(np.sum((eligible==1)&(z>0))),codes=stats,bad=int(np.sum(data[:,10]!=0)),certified=int(np.sum(data[:,0]==1)),unresolved=int(np.sum(data[:,0]==0)),no_candidate=int(np.sum(data[:,0]==2)),first_time=t,times=timings,cache_build_time=build_time,probe_pixels=len(chosen),probe_records=len(ids),anomalies=anomalies,output_bytes=data.nbytes,view_wall_seconds=time.perf_counter()-start)
                    save(out/f'scan{scan}_{name}.json',row);rows.append(row);save(out/'progress.json',dict(utc=now(),completed_views=len(rows),phase='view_complete'));print(f'{len(rows)}/24 scan{scan}/{name}: codes={stats}, bad={row["bad"]}, ms={np.median([x["event_ms"] for x in timings]):.3f}',flush=True)
                    del inputs,initial,src,cache,prosrc,probeout,data,args,records,offsets,query,amp,counters;torch.cuda.empty_cache()
            del scene,gs,cams;torch.cuda.empty_cache()
        assert sha(ply)==expected
    result=dict(status='complete',views=len(rows),rows=rows,totals={k:sum(r[k] for r in rows) for k in ['records','pixels','initial_eligible','original_valid','certified','unresolved','no_candidate','bad','probe_pixels','probe_records']},independent_MP_audit='pending',scope='Sufficient-condition coverage for original F32 depth; no fallback root/backward/GT/training completed.')
    save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h,n
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',state);print(json.dumps(result['totals']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve()
    if out.exists():raise FileExistsError(out)
    sources=[Path(__file__),R/'FLOAT_PRODUCT_PROTOCOL.md',R/'float_product.cuh',R/'float_product_cuda.py',R/'benchmark_csr_decode.py',R/'benchmark_csr_capture.py',R/'benchmark_full_topology.py',R/'benchmark_median_cuda.py',R/'exact_outlier_capture_rebuilt.py',R/'exact_perspective_backend.py',R/'guarded_interval_cuda.py',TOPO/'manifest.json',CSR/'manifest.json',DECODE/'manifest.json',CACHE/'manifest.json',R/'experiments/exact_outlier_capture_v3/summary.json']+list((ROOT/'scene').glob('*.py'))+[CACHE/'generated'/n for n in ['guarded_interval.cuh','guarded_preprocess.cuh','shared_cache.cuh']]
    out.mkdir(parents=True);state=dict(status='building',created_utc=now(),pid=os.getpid(),source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources});save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state)
    except BaseException as e:
        state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
