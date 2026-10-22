"""All previously unresolved actual rays: dynamic low/high strict correction."""
from pathlib import Path
import argparse,os,tempfile,time,traceback,zipfile
import numpy as np
import torch
from research.benchmark_csr_decode import src_rows,read,sha,save,now,event
from research.benchmark_csr_capture import pinned
from research.exact_perspective_backend import ROOT
from research.exact_outlier_capture_rebuilt import native_args
from research.benchmark_median_cuda import load_config
from research.benchmark_full_topology import digest
from research.dynamic_root_cuda import load
R=ROOT/'research';CACHE=R/'experiments/shared_cache_v1';CSR=R/'experiments/csr_capture_v1';DECODE=R/'experiments/csr_decode_v1';FLOAT=R/'experiments/float_product_v1';TOPO=R/'experiments/full_topology_v1'
def compact(pixels,offsets,records,q):
    counts=np.diff(offsets)[pixels];sections=np.concatenate([[0],np.cumsum(counts)]).astype(np.int64);indices=np.concatenate([np.arange(offsets[p],offsets[p+1],dtype=np.int64) for p in pixels]);ids=records[indices,0];query=np.ones((len(pixels),4),np.float64);query[:,:2]=q[pixels];return sections,indices,ids,query
def run(out,state):
    from scene import Scene,GaussianModel
    torch.set_num_threads(1);top=read(TOPO/'manifest.json');cm=read(CACHE/'manifest.json');dm=read(DECODE/'manifest.json');cr=read(CSR/'manifest.json');fm=read(FLOAT/'manifest.json');fs=read(FLOAT/'summary.json');bfile=next(n for n in cm['generated_sha256'] if n.endswith('.so'));meta=dict(binary_relative_path=(CACHE/'generated'/bfile).relative_to(ROOT).as_posix(),binary_sha256=cm['generated_sha256'][bfile]);geometry=pinned(meta);module,backend=load(out);state.update(status='running',backend=backend,cache_backend=meta);save(out/'manifest.json',state);rows=[]
    for scan in [37,55,63]:
        model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=top['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected;ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='dynamic-root-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch);gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False);cams={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()};bg=torch.tensor([1.,1.,1.] if ds.white_background else [0.,0.,0.],device='cuda')
            for vi,oldrow in enumerate(r for r in fs['rows'] if r['scan']==scan):
                name=oldrow['camera'];stem=f'scan{scan}_{name}';view_start=time.perf_counter();save(out/'progress.json',dict(utc=now(),completed_views=len(rows),active_view=stem,phase='prepare'))
                with torch.no_grad():
                    cam=cams[name];args=native_args(cam,gs,pipe,bg,ds.kernel_size)
                    if vi==0:assert {str(i):digest(args[i]) for i in range(11)}==top['model_inputs'][str(scan)]['tensors']
                    assert {str(i):digest(args[i]) for i in [14,15,21]}==read(TOPO/(stem+'.json'))['camera_tensor_inputs']
                    for folder,manifest,filename in [(CSR,cr,stem+'.npz'),(DECODE,dm,stem+'_query_amp.npz'),(FLOAT,fm,stem+'_output.npz'),(FLOAT,fm,stem+'_probes.npz')]:assert sha(folder/filename)==manifest['artifacts_sha256'][filename]
                    with np.load(CSR/(stem+'.npz')) as d:records=d['records'];offsets=d['offsets']
                    with np.load(DECODE/(stem+'_query_amp.npz')) as d:q=d['query_xy'];amp=d['amplitude']
                    with np.load(FLOAT/(stem+'_output.npz')) as d:prior=d['output']
                    with np.load(FLOAT/(stem+'_probes.npz')) as d:oldprobe=d['pixels'][d['output'][:,0]==0]
                    pixels=np.flatnonzero(prior[:,0]==0);assert len(pixels)==oldrow['unresolved'];counts=np.diff(offsets)[pixels];chosen=set(oldprobe.tolist());chosen.update(pixels[np.linspace(0,len(pixels)-1,min(32,len(pixels)),dtype=np.int64)].tolist());chosen=np.array(sorted(chosen),np.int64);src=src_rows(args);cache,cachetime=event(lambda:geometry.build(src));at=torch.tensor(amp,device='cuda');assert bool(torch.all(cache[:,:2]==0).item())
                    cpustart=time.perf_counter();sections,indices,ids,query=compact(pixels,offsets,records,q);compact_seconds=time.perf_counter()-cpustart;query[:,3]=prior[pixels,11]
                    final=np.full((len(pixels),20),np.nan,np.float64);probeids=np.searchsorted(pixels,chosen);probe_counts=counts[probeids];prosections=np.concatenate([[0],np.cumsum(probe_counts)]);probe_indices=np.concatenate([np.arange(sections[k],sections[k+1],dtype=np.int64) for k in probeids]);prosrc=src[torch.tensor(ids[probe_indices],device='cuda',dtype=torch.long)].clone();prosrc[:,32:34]=torch.tensor(q[np.repeat(chosen,probe_counts)],device='cuda',dtype=torch.float64);prosrc[:,34]=1;prosrc[:,35]=torch.tensor(amp[ids[probe_indices]],device='cuda',dtype=torch.float64);probe_packets=np.full((len(probe_indices),26),np.nan,np.float64);modes=[];active=np.arange(len(pixels));torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
                    for mode in [0,1]:
                        if len(active)==0:break
                        if mode:
                            cpustart=time.perf_counter();sections,indices,ids,query=compact(pixels[active],offsets,records,q);query[:,3]=prior[pixels[active],11];compact_seconds+=time.perf_counter()-cpustart
                        st=torch.tensor(sections,device='cuda');it=torch.tensor(ids,dtype=torch.int32,device='cuda');qt=torch.tensor(query,device='cuda');save(out/'progress.json',dict(utc=now(),completed_views=len(rows),active_view=stem,phase='decode',mode=mode,active_rays=len(active),active_nodes=len(ids)))
                        packet,dt=event(lambda:module.decode(cache,it,st,qt,at,mode));save(out/'progress.json',dict(utc=now(),completed_views=len(rows),active_view=stem,phase='solve',mode=mode));root,rt=event(lambda:module.solve(packet,st,qt,mode));data=root.cpu().numpy();times=[]
                        for repeat in range(2):
                            value,tm=event(lambda:module.solve(packet,st,qt,mode));assert np.array_equal(value.cpu().numpy().view(np.uint64),data.view(np.uint64));times.append(tm);del value
                        final[active]=data
                        # Entire selected ray packets copied according to active dynamic sections.
                        pixel_active=pixels[active];lookup={int(p):j for j,p in enumerate(pixel_active)}
                        for j,p in enumerate(chosen):
                            if int(p) in lookup:
                                k=lookup[int(p)];probe_packets[int(prosections[j]):int(prosections[j+1])]=packet[int(sections[k]):int(sections[k+1])].cpu().numpy()
                        cu,cn=np.unique(data[:,0],return_counts=True);modes.append(dict(mode=mode,rays=len(active),records=len(ids),decode_time=dt,first_root_time=rt,repeat_root_times=times,codes={str(int(k)):int(v) for k,v in zip(cu,cn)},packet_bytes=packet.numel()*packet.element_size()))
                        active=active[data[:,0]==0];del packet,root,st,it,qt,data;torch.cuda.empty_cache()
                    assert not np.isnan(final[:,0]).any();cu,cn=np.unique(final[:,0],return_counts=True);methods={str(k):int(np.sum((final[:,0]==1)&(final[:,12]==k))) for k in [1,2,3]};max_added=torch.cuda.max_memory_allocated()-before
                    np.savez_compressed(out/(stem+'_output.npz'),pixels=pixels,output=final)
                    np.savez_compressed(out/(stem+'_probes.npz'),pixels=chosen,sections=prosections,source=prosrc.cpu().numpy(),packets=probe_packets,output=final[probeids],record_indices=np.concatenate([np.arange(offsets[p],offsets[p+1],dtype=np.int64) for p in chosen]),primitive_ids=records[np.concatenate([np.arange(offsets[p],offsets[p+1],dtype=np.int64) for p in chosen]),0],query=q[chosen])
                    row=dict(scan=scan,camera=name,width=cam.image_width,height=cam.image_height,selected_rays=len(pixels),selected_records=int(counts.sum()),counts_above192=int(np.sum(counts>192)),max_contributors=int(counts.max()),final_codes={str(int(k)):int(v) for k,v in zip(cu,cn)},certificates=methods,finite_low=int(np.sum((final[:,0]==1)&(final[:,1]==0))),finite_high=int(np.sum((final[:,0]==1)&(final[:,1]==1))),certified=int(np.sum(final[:,0]==1)),unresolved=int(np.sum(final[:,0]==0)),no_finite_root=int(np.sum((final[:,0]==2)|(final[:,0]==3))),modes=modes,cache_build_time=cachetime,CPU_compact_seconds=compact_seconds,max_added_allocated_bytes=max_added,probe_rays=len(chosen),probe_records=len(probe_indices),preserved_original_certified=int(np.sum(prior[:,0]==1)),original_anomalies=oldrow['anomalies'],view_wall_seconds=time.perf_counter()-view_start)
                    save(out/(stem+'.json'),row);rows.append(row);print(f'{len(rows)}/24 {stem}: codes={row["final_codes"]}, methods={methods}, low/high={row["finite_low"]}/{row["finite_high"]}',flush=True);save(out/'progress.json',dict(utc=now(),completed_views=len(rows),phase='view_complete'))
                    del args,records,offsets,q,amp,prior,src,cache,at,prosrc,probe_packets,final;torch.cuda.empty_cache()
            del scene,gs,cams;torch.cuda.empty_cache()
        assert sha(ply)==expected
    result=dict(status='complete',rows=rows,views=len(rows),totals={k:sum(r[k] for r in rows) for k in ['selected_rays','selected_records','counts_above192','certified','unresolved','no_finite_root','finite_low','finite_high','probe_rays','probe_records','preserved_original_certified']},independent_world_MP_audit='pending',scope='All previously unresolved original-valid development pixels, strict sufficient certificates only; no production/backward/GT/efficiency/training adoption.')
    assert result['totals']['selected_rays']==562335 and result['totals']['preserved_original_certified']==8583586;save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h,n
    state.update(status='complete',completed_utc=now(),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ['manifest.json','progress.json']});save(out/'manifest.json',state)
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True)
    sources=[Path(__file__),R/'dynamic_root.cuh',R/'dynamic_root_cuda.py',R/'DYNAMIC_ROOT_PROTOCOL.md',R/'float_product.cuh',R/'float_product_cuda.py',R/'csr_decode_cuda.py',R/'benchmark_csr_decode.py',R/'benchmark_csr_capture.py',R/'benchmark_full_topology.py',R/'benchmark_median_cuda.py',R/'exact_outlier_capture_rebuilt.py',R/'exact_perspective_backend.py',R/'guarded_interval_cuda.py',TOPO/'manifest.json',CSR/'manifest.json',DECODE/'manifest.json',FLOAT/'manifest.json',FLOAT/'summary.json',CACHE/'manifest.json']+list((ROOT/'scene').glob('*.py'))+[CACHE/'generated'/n for n in ['guarded_preprocess.cuh','shared_cache.cuh']]+[R/'experiments/short_certificate_v1/variant_generated'/n for n in ['guarded_interval.cuh','short_interval.cuh']]
    state=dict(status='building',created_utc=now(),pid=os.getpid(),source_sha256={s.relative_to(ROOT).as_posix():sha(s) for s in sources});save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for s in sources:z.write(s,s.relative_to(ROOT).as_posix())
    try:run(out,state)
    except BaseException as e:state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
