"""Full49-view frozen-model original/certified depth TSDF and seeded DTU gate."""
from pathlib import Path
import argparse,gc,json,os,subprocess,sys,tempfile,time,traceback,zipfile
import numpy as np
import torch
import open3d as o3d
from research.benchmark_csr_decode import src_rows,read,sha,save,now,event
from research.benchmark_csr_capture import pinned,offsets_for
from research.benchmark_full_topology import digest
from research.benchmark_median_cuda import load_config
from research.exact_outlier_capture_rebuilt import native_args,parse
from research.exact_perspective_backend import ROOT
R=ROOT/'research';E=R/'experiments';TOPO=E/'full_topology_v1'

def pinned_any(meta):
    info=dict(binary_sha256=meta['binary_sha256'])
    info['binary_relative_path']=meta.get('binary_relative_path') or Path(meta['binary_path']).relative_to(ROOT).as_posix()
    return pinned(info),info

def same(a,b):
    return bool(a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.int32),b.view(torch.int32)))

def compact(pixels,offsets,records,query,z):
    counts=offsets[pixels+1]-offsets[pixels];sections=torch.cat([torch.zeros(1,dtype=torch.int64,device='cuda'),torch.cumsum(counts,0)]);L=int(sections[-1].item());owners=torch.repeat_interleave(torch.arange(len(pixels),device='cuda'),counts,output_size=L);index=offsets[pixels][owners]+torch.arange(L,device='cuda')-sections[owners];ids=records[index,0].contiguous();q=torch.ones((len(pixels),4),device='cuda',dtype=torch.float64);q[:,:2]=query[pixels].double();q[:,3]=z[pixels].double();return ids,sections,q

def correct(root,cache,records,offsets,query,amp,z,pixels):
    final=torch.full((len(pixels),20),float('nan'),dtype=torch.float64,device='cuda');active=torch.arange(len(pixels),device='cuda');stages=[]
    for mode in [0,1]:
        if active.numel()==0:break
        inputs,ct=event(lambda:compact(pixels[active],offsets,records,query,z));ids,sections,q=inputs
        packet,dt=event(lambda:root.decode(cache,ids,sections,q,amp,mode));out,rt=event(lambda:root.solve(packet,sections,q,mode));final[active]=out;active=active[out[:,0]==0];stages.append(dict(mode=mode,rays=int(len(out)),records=int(len(ids)),compact=ct,decode=dt,root=rt));del inputs,ids,sections,q,packet,out
    return final,stages

def volume():
    return o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),attr_dtypes=(o3d.core.float32,)*3,attr_channels=(1,1,3),voxel_size=.002,block_resolution=16,block_count=50000,device=o3d.core.Device('CPU:0'))

def integrate(vbg,rgb,depth,intrinsic,extrinsic):
    d=o3d.t.geometry.Image(np.ascontiguousarray(depth));c=o3d.t.geometry.Image(np.ascontiguousarray(rgb));K=o3d.core.Tensor(intrinsic);X=o3d.core.Tensor(extrinsic);blocks=vbg.compute_unique_block_coordinates(d,K,X,1.,8.,4.);vbg.integrate(blocks,d,c,K,X,1.,8.,4.)

def alignment(cameras):
    from evaluate_dtu_mesh import load_dtu_camera,best_fit_transform
    names=[c.image_name for c in cameras];assert names==[f'{i:04d}' for i in range(49)],names
    official=np.array([p[:,3] for p in load_dtu_camera(str(ROOT/'dtu_eval/Offical_DTU_Dataset'))],np.float64);indices=[int(n) for n in names];truth=official[indices];centers=np.array([torch.linalg.inv(c.world_view_transform.T)[:3,3].cpu().numpy() for c in cameras]);scale=np.linalg.norm(truth-truth.mean(0),axis=1).mean()/np.linalg.norm(centers-centers.mean(0),axis=1).mean();_,rot,tr=best_fit_transform(centers*scale,truth)
    return dict(scale=float(scale),rotation=rot.tolist(),translation=tr.tolist(),camera_names=names,official_indices=indices,source_camera_centers=centers.tolist(),official_camera_centers=truth.tolist())

def write_mesh(vbg,path,align):
    from mesh_extract import post_process_mesh
    start=time.perf_counter();mesh=vbg.extract_triangle_mesh();mesh.compute_vertex_normals();legacy=mesh.to_legacy();assert len(legacy.vertices)>0 and len(legacy.triangles)>0;o3d.io.write_triangle_mesh(str(path/'recon.ply'),legacy);processed=post_process_mesh(legacy,1);o3d.io.write_triangle_mesh(str(path/'recon_post.ply'),processed);vertices=np.asarray(processed.vertices);vertices[:]=vertices*align['scale']@np.asarray(align['rotation']).T+np.asarray(align['translation']);processed.compute_vertex_normals();o3d.io.write_triangle_mesh(str(path/'recon_aligned.ply'),processed)
    return dict(vertices=len(processed.vertices),triangles=len(processed.triangles),wall_seconds=time.perf_counter()-start,files_sha256={p.name:sha(p) for p in path.glob('recon*.ply')})

def evaluate(out,scan,arm,state,rootout):
    metrics=[]
    for seed in [0,1,2]:
        dest=out/f'eval_{arm}_seed{seed}';log=out/f'eval_{arm}_seed{seed}.log';cmd=[sys.executable,'-m','research.evaluate_mesh_seeded','--mesh',str(out/arm/'recon_aligned.ply'),'--scan',str(scan),'--seed',str(seed),'--workers','2','--out',str(dest)];entry=dict(scan=scan,arm=arm,seed=seed,status='running',argv=cmd);state.setdefault('evaluation_steps',[]).append(entry);save(rootout/'manifest.json',state)
        with log.open('w',encoding='utf-8') as f:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);entry['pid']=process.pid;save(rootout/'manifest.json',state)
            while process.poll() is None:
                save(rootout/'progress.json',dict(utc=now(),phase='evaluating',scan=scan,arm=arm,seed=seed,pid=process.pid));time.sleep(10)
            rc=process.returncode
        entry.update(status='complete' if rc==0 else 'failed',exit_code=rc);save(rootout/'manifest.json',state);assert rc==0,(cmd,log);metrics.append(read(dest/'results.json'));print(f'scan{scan} {arm} seed{seed}: {metrics[-1]}',flush=True)
    return metrics

def run(out,state):
    from scene import Scene,GaussianModel
    from gaussian_renderer import render
    import diff_gaussian_rasterization
    torch.set_num_threads(1);top=read(TOPO/'manifest.json');mods={};backends={}
    for label,folder,key in [('query','csr_decode_v1','capture_backend'),('csr','csr_capture_v1','csr_backend'),('certificate','float_product_v1','backend'),('root','dynamic_root_v1','backend')]:mods[label],backends[label]=pinned_any(read(E/folder/'manifest.json')[key])
    cm=read(E/'shared_cache_v1/manifest.json');bfile=next(n for n in cm['generated_sha256'] if n.endswith('.so'));mods['geometry'],backends['geometry']=pinned_any(dict(binary_relative_path=(E/'shared_cache_v1/generated'/bfile).relative_to(ROOT).as_posix(),binary_sha256=cm['generated_sha256'][bfile]));original_binary=Path(diff_gaussian_rasterization._C.__file__);state.update(status='rendering',backends=backends,original_renderer_binary=dict(path=str(original_binary),sha256=sha(original_binary)));save(out/'manifest.json',state);scene_rows=[]
    for scan in [37,55,63]:
        scanstart=time.perf_counter();model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ply=model/'point_cloud/iteration_30000/point_cloud.ply';expected=top['model_inputs'][str(scan)]['ply_sha256'];assert sha(ply)==expected;dest=out/f'scan{scan}';dest.mkdir();(dest/'control').mkdir();(dest/'certified').mkdir();ds,pipe=load_config(model);rows=[];control_vbg=volume();candidate_vbg=volume()
        with tempfile.TemporaryDirectory(prefix='frozen-quality-scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch);gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False);cameras=scene.getTrainCameras();assert len(cameras)==49 and len(scene.getTestCameras())==0;align=alignment(cameras);save(dest/'alignment.json',align);background=torch.ones(3,device='cuda',dtype=torch.float32)
            for vi,cam in enumerate(cameras):
                start=time.perf_counter();stem=cam.image_name;save(out/'progress.json',dict(utc=now(),phase='original_forward',scan=scan,camera=stem,completed_in_scene=len(rows)))
                with torch.no_grad():
                    original,t_orig=event(lambda:render(cam,gs,pipe,background,ds.kernel_size));args=native_args(cam,gs,pipe,background,ds.kernel_size);P=len(args[1]);W,H=cam.image_width,cam.image_height
                    n,t_query=event(lambda:mods['query'].rasterize_gaussians(*args));checks={k:same(original[a],n[b]) for k,a,b in [('rgb','render',1),('alpha','mask',2),('normal','normal',3),('radii','radii',5)]};checks['depth']=same(original['median_depth'],n[4][:1]);checks['slope']=same(original['interface_slope'],n[4][1:2]);assert all(checks.values()),checks
                    counts=n[4][3].to(torch.int64);(offsets,records,status),t_alloc=event(lambda:offsets_for(counts));capture,t_capture=event(lambda:mods['csr'].rasterize_gaussians(*args,offsets,records,status));assert same(capture[1],n[1]) and same(capture[4],n[4][:6]) and bool(torch.all(status==0).item());parsed,_=parse(capture,'affine',P,W,H);checked,t_check=event(lambda:mods['csr'].validate_csr(records,offsets,parsed['tile']['ranges'],parsed['binning']['point_list'],parsed['image']['n_contrib'],W,H,P));assert bool(torch.all(checked==0).item())
                    qp,_=parse(n,'affine',P,W,H);amp=qp['geom']['conic'][:,3].contiguous();query=n[4][6:8].permute(1,2,0).reshape(-1,2).contiguous();z=n[4][0].reshape(-1).contiguous();eligible=n[4][5].reshape(-1).contiguous();source=src_rows(args);cache,t_cache=event(lambda:mods['geometry'].build(source));assert bool(torch.all(cache[:,:2]==0).item()) and bool(torch.isfinite(cache).all().item())
                    certificate,t_cert=event(lambda:mods['certificate'].run(cache,records,offsets,query,amp,z,eligible));pixels=torch.nonzero(certificate[:,0]==0).reshape(-1);save(out/'progress.json',dict(utc=now(),phase='correcting',scan=scan,camera=stem,selected_rays=len(pixels),records=len(records)))
                    if len(pixels):roots,root_stages=correct(mods['root'],cache,records,offsets,query,amp,z,pixels)
                    else:roots=torch.empty((0,20),device='cuda',dtype=torch.float64);root_stages=[]
                    newz=z.clone();accepted=roots[:,0]==1;newz[pixels]=0;newz[pixels[accepted]]=roots[accepted,2].float();assert bool(torch.all(torch.isfinite(newz)).item()) and bool(torch.all(newz>=0).item());preserved=certificate[:,0]!=0;assert same(newz[preserved],z[preserved]);assert bool(torch.all(newz[pixels[accepted]]>0).item())
                    control=z.reshape(1,H,W).clone();candidate=newz.reshape(1,H,W).clone();mask=torch.ones((1,H,W),device='cuda',dtype=torch.bool) if cam.gt_mask is None else cam.gt_mask>=.5;control[~mask]=0;candidate[~mask]=0;rgb=original['render'].clamp(0,1).permute(1,2,0).contiguous().cpu().numpy();cdepth=control[0].cpu().numpy();ndepth=candidate[0].cpu().numpy();K=np.array([[cam.Fx,0,cam.Cx],[0,cam.Fy,cam.Cy],[0,0,1]],np.float64);X=cam.world_view_transform.T.cpu().numpy().astype(np.float64)
                    cpu=time.perf_counter();input_hashes=dict(args={str(k):digest(args[k]) for k in [1,3,4,5,14,15,21]},actual_query=digest(query),actual_amp=digest(amp),records=digest(records),offsets=digest(offsets));hash_wall=time.perf_counter()-cpu
                    np.savez_compressed(dest/(stem+'_images.npz'),rgb=rgb,control_depth=cdepth,certified_depth=ndepth,mask=mask.cpu().numpy(),intrinsic=K,extrinsic=X,selected_pixels=pixels.cpu().numpy(),root_output=roots.cpu().numpy(),certificate=certificate.cpu().numpy())
                    record=dict(scan=scan,camera=stem,width=W,height=H,pixels=W*H,records=len(records),original_renderer_bits=checks,CSR_write_checker_zero=True,source_cache_all_valid=True,original_valid=int((z>0).sum().item()),original_certified=int((certificate[:,0]==1).sum().item()),selected_rays=len(pixels),corrected=int(accepted.sum().item()),no_finite_root=int(((roots[:,0]==2)|(roots[:,0]==3)).sum().item()),unresolved=int((roots[:,0]==0).sum().item()),masked_control_valid=int((control>0).sum().item()),masked_certified_valid=int((candidate>0).sum().item()),unchanged_pixels_preserved=True,input_hashes=input_hashes,times=dict(original=t_orig,query=t_query,CSR_allocation=t_alloc,capture=t_capture,checker=t_check,cache=t_cache,certificate=t_cert,root_stages=root_stages,CPU_input_hash_seconds=hash_wall),before_TSDF_wall_seconds=time.perf_counter()-start)
                    del original,n,capture,parsed,qp,args,counts,offsets,records,status,checked,source,cache,amp,query,z,eligible,certificate,pixels,roots,newz,accepted,preserved,control,candidate,mask;torch.cuda.empty_cache()
                save(out/'progress.json',dict(utc=now(),phase='TSDF_integrating',scan=scan,camera=stem));t=time.perf_counter();integrate(control_vbg,rgb,cdepth,K,X);record['control_integration_seconds']=time.perf_counter()-t;t=time.perf_counter();integrate(candidate_vbg,rgb,ndepth,K,X);record['certified_integration_seconds']=time.perf_counter()-t;record['view_wall_seconds']=time.perf_counter()-start;save(dest/(stem+'.json'),record);rows.append(record);print(f'scan{scan} {vi+1}/49 {stem}: certified={record["original_certified"]}+{record["corrected"]}, no-root={record["no_finite_root"]}, unresolved={record["unresolved"]}',flush=True);del rgb,cdepth,ndepth;gc.collect()
            del scene,gs,cameras;torch.cuda.empty_cache()
        save(out/'progress.json',dict(utc=now(),phase='extracting_mesh',scan=scan));meshrows={arm:write_mesh(vbg,dest/arm,align) for arm,vbg in [('control',control_vbg),('certified',candidate_vbg)]};del control_vbg,candidate_vbg;gc.collect();save(dest/'render_summary.json',dict(status='complete',views=49,rows=rows,meshes=meshrows,PLY_sha256=expected));assert sha(ply)==expected
        metrics={arm:evaluate(dest,scan,arm,state,out) for arm in ['control','certified']};means={arm:{k:float(np.mean([v[k] for v in metrics[arm]])) for k in ['overall','mean_d2s','mean_s2d']} for arm in metrics};result=dict(scan=scan,views=49,metrics=metrics,means=means,delta={k:means['certified'][k]-means['control'][k] for k in means['control']},overall_ratio=means['certified']['overall']/means['control']['overall'],RGB_bits_preserved=True,selected_rays=sum(r['selected_rays'] for r in rows),corrected=sum(r['corrected'] for r in rows),no_finite_root=sum(r['no_finite_root'] for r in rows),unresolved=sum(r['unresolved'] for r in rows),masked_control_valid=sum(r['masked_control_valid'] for r in rows),masked_certified_valid=sum(r['masked_certified_valid'] for r in rows),scene_wall_seconds=time.perf_counter()-scanstart);save(dest/'results.json',result);scene_rows.append(result);state['completed_scenes']=len(scene_rows);save(out/'manifest.json',state)
    ratio=float(np.mean([r['means']['certified']['overall'] for r in scene_rows])/np.mean([r['means']['control']['overall'] for r in scene_rows]));gate=ratio<=.99 and all(r['delta']['mean_s2d']<=0 and r['RGB_bits_preserved'] for r in scene_rows);result=dict(status='complete',scenes=scene_rows,views=147,mean_overall_ratio=ratio,quality_gate_pass=gate,scope='Frozen previously used development models; evaluation seed repetitions not independent training. Full CUDA backward, training and newness not established.');save(out/'summary.json',result)
    for n,h in state['source_sha256'].items():assert sha(ROOT/n)==h,n
    state.update(status='complete',completed_utc=now(),quality_gate_pass=gate,summary_sha256=sha(out/'summary.json'));save(out/'manifest.json',state);print(json.dumps({k:v for k,v in result.items() if k!='scenes'},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);out=p.parse_args().out.resolve();assert not out.exists();out.mkdir(parents=True)
    files=[Path(__file__),R/'FROZEN_CERTIFIED_MESH_PROTOCOL.md',R/'benchmark_csr_decode.py',R/'benchmark_csr_capture.py',R/'benchmark_full_topology.py',R/'benchmark_median_cuda.py',R/'exact_outlier_capture_rebuilt.py',R/'exact_perspective_backend.py',R/'evaluate_mesh_seeded.py',ROOT/'gaussian_renderer/__init__.py',ROOT/'mesh_extract.py',ROOT/'evaluate_dtu_mesh.py',ROOT/'dtu_eval/eval.py']+list((ROOT/'scene').glob('*.py'))+[E/x/'manifest.json' for x in ['csr_decode_v1','csr_capture_v1','shared_cache_v1','float_product_v1','dynamic_root_v1']]
    gt=ROOT/'dtu_eval/Offical_DTU_Dataset'
    for scan in [37,55,63]:files+=[gt/'ObsMask'/f'ObsMask{scan}_10.mat',gt/'ObsMask'/f'Plane{scan}.mat',gt/'Points/stl'/f'stl{scan:03d}_total.ply']
    files+=list((gt/'Calibration').rglob('pos_*.txt'));state=dict(status='starting',created_utc=now(),pid=os.getpid(),source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in files});save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:
            if not str(f).startswith(str(gt)):z.write(f,f.relative_to(ROOT).as_posix())
    try:run(out,state)
    except BaseException as e:state.update(status='failed',failed_utc=now(),failure_type=type(e).__name__,failure=str(e));save(out/'manifest.json',state);(out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
