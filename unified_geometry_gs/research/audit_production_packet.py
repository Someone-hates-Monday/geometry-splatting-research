"""Frozen real-row audit with copied full production FP32 CUDA preprocessing."""
from __future__ import annotations
import argparse,hashlib,json,math,os,platform,subprocess,tempfile,time,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from research.clipped_packet import recover
from research.clipped_packet_torch import recover as cuda_recover
from research.production_packet_backend import load,RASTER,FLAGS
from research.benchmark_median_cuda import load_config

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def command(args):
    try:return subprocess.run(args,check=False,capture_output=True,text=True).stdout.strip()
    except OSError as e:return str(e)
def evaluate(candidate,target,diag,culled):
    difference=abs(candidate-target)
    errors=np.column_stack((difference[:,0]/np.maximum(1e-8,abs(target[:,0])),
        difference[:,1]/np.maximum(1e-8,abs(target[:,1])),difference[:,2]))
    invalid=(~np.all(np.isfinite(candidate),axis=1))|(~np.isfinite(diag['energy']))|(diag['energy']<0)|(diag['detK']<=0)|(diag['denominator']<=0)
    bad=culled|invalid|np.any(errors>np.array([1e-5,1e-4,1e-4]),axis=1)
    finite=np.all(np.isfinite(errors),axis=1)
    maxerrors=np.max(errors[finite],axis=0).tolist() if np.any(finite) else None
    return errors,bad,{'failed_inputs':int(sum(bad)),'invalid_inputs':int(sum(invalid)),
        'finite_error_inputs':int(sum(finite)),'max_errors_finite_only':maxerrors}

def decode(values,dtype):
    p,w,a,d=recover(*[np.asarray(v,dtype=dtype) for v in values]);return np.column_stack((p,w,a)),d

def execute(out,state,previous):
    from scene import GaussianModel,Scene
    torch.set_num_threads(1);mod=load(out);torch.cuda.synchronize()
    state.update(status='running',binary_sha256=sha(Path(mod.__file__)),binary_path=str(mod.__file__))
    state['build_ninja_sha256']=sha(out/'generated/build.ninja');save(out/'manifest.json',state)
    rows=[];start=time.perf_counter()
    for scan in (37,55,63):
        model=ROOT/f'output_compare/ggs_pipeline/scan{scan}';ds,pipe=load_config(model)
        with tempfile.TemporaryDirectory(prefix='scene-',dir=out) as temp:
            scratch=Path(temp);(scratch/'point_cloud').symlink_to(model/'point_cloud',target_is_directory=True);ds.model_path=str(scratch)
            gs=GaussianModel(ds.sh_degree,ds.sg_degree);scene=Scene(ds,gs,load_iteration=30000,shuffle=False)
            cameras={c.image_name:c for c in scene.getTrainCameras()+scene.getTestCameras()}
            scale,opacity=gs.get_scaling_n_opacity_with_3D_filter
            quaternion=gs.get_rotation
            for _ in range(8):
                i=len(rows);prior=read(previous/f'packet_{i:02d}.json');assert prior['scan']==scan
                cam=cameras[prior['camera']];W,H=cam.image_width,cam.image_height
                focal=np.array([cam.Fx,cam.Fy]);center=np.array([cam.Cx,cam.Cy]);tan=np.array([math.tan(cam.FoVx/2),math.tan(cam.FoVy/2)])
                with np.load(previous/f'packet_{i:02d}.npz',allow_pickle=False) as packet:
                    ids=packet['selected_ids'];idx=torch.tensor(ids.astype(np.int64),device='cuda');target=packet['expected']
                    assert np.max(abs(packet['focal']-focal))<1e-9
                    m=gs.get_xyz[idx].detach().contiguous();s=scale[idx].detach().contiguous();q=quaternion[idx].detach().contiguous();o=opacity[idx].detach().flatten().contiguous()
                    v=cam.world_view_transform.detach().contiguous();p=cam.full_proj_transform.detach().contiguous();origin=cam.camera_center.detach().contiguous()
                    result=mod.forward(m,s,q,o,v,p,origin,W,H,*tan,*focal);torch.cuda.synchronize()
                    plane,conic,uv,radii,depth,touched,geom=[x.detach().cpu().numpy() for x in result]
                    culled=(radii<=0)|(touched<=0)
                    with np.errstate(all='ignore'):
                        # Geometry consumed by a prospective packet-only CUDA implementation.
                        f32=focal.astype(np.float32);c32=center.astype(np.float32);limit=(np.float32(1.3)*tan.astype(np.float32))
                        uvmean=np.column_stack(((uv-c32)/f32,np.ones(len(ids),dtype=np.float32)))
                        uvbar=np.column_stack((np.clip(uvmean[:,:2],-limit,limit),np.ones(len(ids),dtype=np.float32)))
                        query=np.column_stack((np.broadcast_to((np.asarray(prior['pixel'],dtype=np.float32)-c32)/f32,(len(ids),2)),np.ones(len(ids),dtype=np.float32)))
                        width=1/plane[:,3];values=[plane[:,:3],width,conic[:,:3],uvmean,uvbar,query,f32,conic[:,3]]
                        candidate64,diag64=decode(values,np.float64)
                        candidate32,diag32=decode(values,np.float32)
                        gpuvalues=[torch.tensor(x,dtype=torch.float32,device='cuda') for x in values]
                        candidatecuda=cuda_recover(*gpuvalues).detach().cpu().numpy()
                        camvalues=list(values);camvalues[3]=geom[:,:3];camvalues[4]=geom[:,3:6]
                        cameracandidate,cameradiag=decode(camvalues,np.float64)
                        exactvalues=list(values);exactvalues[3]=packet['qmean'];exactvalues[4]=packet['qbar'];exactvalues[5]=packet['qray'];exactvalues[6]=packet['focal']
                        exactcandidate,exactdiag=decode(exactvalues,np.float64)
                    arms={};arrays={}
                    for name,candidate,diag in (('production_uv_fp64_decode',candidate64,diag64),('production_uv_numpy_fp32_decode',candidate32,diag32),
                            ('production_uv_cuda_fp32_decode',candidatecuda,diag32),('extra_camera_ratio_fp64_decode',cameracandidate,cameradiag),
                            ('archived_double_geometry_fp64_decode',exactcandidate,exactdiag)):
                        error,bad,stat=evaluate(candidate,target,diag,culled);stat['failed_ids']=ids[bad].tolist();arms[name]=stat
                        arrays[name+'_candidate']=candidate;arrays[name+'_errors']=error;arrays[name+'_bad']=bad
                    arrays.update(selected_ids=ids,clipped=packet['clipped'],expected=target,plane=plane,conic_opacity=conic,uv=uv,radii=radii,depth=depth,tiles_touched=touched,
                        diagnostic_geometry=geom,qmean_uv=uvmean,qbar_uv=uvbar,qray=query,focal=f32,world_view_transform=v.cpu().numpy(),full_proj_transform=p.cpu().numpy(),
                        normalized_quaternion=q.cpu().numpy(),filtered_scale=s.cpu().numpy(),filtered_opacity=o.cpu().numpy(),means=m.cpu().numpy(),
                        archived_plane=packet['plane'],archived_width=packet['width'],archived_conic=packet['conic'],archived_qmean=packet['qmean'],archived_qbar=packet['qbar'])
                    np.savez_compressed(out/f'point_{i:02d}.npz',**arrays)
                    finite_uv=np.all(np.isfinite(uvmean),axis=1)
                    row={'scan':scan,'camera':prior['camera'],'pixel':prior['pixel'],'inputs':len(ids),'clipped_inputs':int(sum(packet['clipped'])),
                        'culled_inputs':int(sum(culled)),'culled_ids':ids[culled].tolist(),'arms':arms,
                        'max_uv_vs_probe_qmean_absolute_finite':float(np.max(abs(uvmean[finite_uv]-geom[finite_uv,:3]))),
                        'max_uv_vs_archived_qmean_absolute_finite':float(np.max(abs(uvmean[finite_uv]-packet['qmean'][finite_uv])))}
                rows.append(row);save(out/f'point_{i:02d}.json',row);save(out/'progress.json',{'utc':now(),'completed':len(rows),'last_row':row})
                print(f'{len(rows)}/24: cull={sum(culled)}, FP64decodefail={arms["production_uv_fp64_decode"]["failed_inputs"]}, CUDAdecodefail={arms["production_uv_cuda_fp32_decode"]["failed_inputs"]}',flush=True)
            del scene,gs,scale,opacity,quaternion;torch.cuda.empty_cache()
    names=list(rows[0]['arms']);summary={'inputs':sum(r['inputs'] for r in rows),'clipped_inputs':sum(r['clipped_inputs'] for r in rows),
        'culled_inputs':sum(r['culled_inputs'] for r in rows),'arms':{name:{'failed_inputs':sum(r['arms'][name]['failed_inputs'] for r in rows),
        'invalid_inputs':sum(r['arms'][name]['invalid_inputs'] for r in rows),'max_errors_finite_only':np.max([r['arms'][name]['max_errors_finite_only'] for r in rows],axis=0).tolist()} for name in names},
        'rows':rows,'execution_scope':'Copied verbatim production CUDA preprocessing functions directly launched in a diagnostic extension; not installed renderer binary.',
        'limits':['No backward/shared primitive accumulation, full renderer, quality, efficiency or novelty claim.',
        'NumPy/CUDA screen coordinate decode precision and production projection epsilon are included.',
        'Extra geometry arms consume information absent from the production packet.',
        'CUDA recovery uses separate Torch FP32 kernels, not a fused production implementation.',
        'Same source functions and production fast-math flags do not prove identical instruction/FMA choices to installed module.']}
    summary['development_gate_pass']=summary['arms']['production_uv_fp64_decode']['failed_inputs']==0 and summary['arms']['production_uv_cuda_fp32_decode']['failed_inputs']==0
    save(out/'summary.json',summary)
    state.update(status='complete',completed_utc=now(),execution_seconds=time.perf_counter()-start,development_gate_pass=summary['development_gate_pass'],
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('manifest.json','progress.json')},
        generated_sha256={p.name:sha(p) for p in (out/'generated').iterdir() if p.is_file()})
    save(out/'manifest.json',state);print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
    if out.exists():raise ValueError('Fresh output required; do not restart after observation timeout.')
    previous=ROOT/'research/experiments/clipped_packet_v1';prior=read(previous/'manifest.json');assert prior['status']=='complete'
    for name,digest in prior['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in prior['artifacts_sha256'].items():assert sha(previous/name)==digest,name
    sources={ROOT/name for name in prior['source_sha256']}
    sources.update([Path(__file__),ROOT/'research/production_packet_backend.py',ROOT/'research/PRODUCTION_PACKET_PROTOCOL.md',ROOT/'research/clipped_packet_torch.py',
        ROOT/'research/benchmark_median_cuda.py',ROOT/'scene/cameras.py',ROOT/'scene/gaussian_model.py',ROOT/'utils/graphics_utils.py',RASTER/'setup.py',RASTER/'cuda_rasterizer/render_forward.cu'])
    sources.update((RASTER/'cuda_rasterizer').glob('*.h'));sources.update(x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file())
    sources.update([previous/'manifest.json',previous/'summary.json'])
    for i in range(24):sources.update([previous/f'packet_{i:02d}.json',previous/f'packet_{i:02d}.npz'])
    out.mkdir(parents=True);state={'created_utc':now(),'status':'building','pid':os.getpid(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(),'python':platform.python_version(),'nvcc':command(['/usr/local/cuda/bin/nvcc','--version']),
        'gcc':command(['gcc','--version']),'nvcc_flags':FLAGS,'production_source_unchanged':True,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}}
    save(out/'manifest.json',state)
    with zipfile.ZipFile(out/'sources.zip','x',zipfile.ZIP_DEFLATED) as z:
        for source in sorted(sources):z.writestr(str(source.relative_to(ROOT)),source.read_bytes())
    try:execute(out,state,previous)
    except BaseException as exc:
        state.update(status='failed',failed_utc=now(),failure_type=type(exc).__name__,failure=str(exc));save(out/'manifest.json',state)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
