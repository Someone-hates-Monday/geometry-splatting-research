"""Observe all actual sorted pairs on unchanged oracle traversal, with endpoint parity."""
import hashlib,importlib.util,json,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_real_work_v7';V5=ROOT/'experiments/tgs_phase_cost_v5'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
def align(n):return (n+127)//128*128
def views(r,n,w,h):
    raw=r[4];pos=0;field={}
    for name,size in [('depth',4),('clamped',3),('radii',4),('anchor',8),('conic',16),('phi_center',8),('rgb',12)]:pos=align(pos);field[name]=pos;pos+=size*n
    maxoff=raw.numel()-128-8*n;minoff=maxoff-align(8*n);countoff=minoff-align(12*n);erroroff=countoff-align(40*n);coeffoff=erroroff-align(80*n)
    def g(off,dtype,width):return raw.narrow(0,off,n*width*torch.empty((),dtype=dtype).element_size()).view(dtype).reshape(n,width)
    hw=w*h;rangeoff=align(align(12*hw)+8*hw);tiles=((w+15)//16)*((h+15)//16)
    return dict(ranges=r[6].narrow(0,rangeoff,8*tiles).view(torch.int32).reshape(tiles,2),ids=r[5].narrow(0,0,4*int(r[0])).view(torch.int32),coeff=g(coeffoff,torch.float64,10),anchor=g(field['anchor'],torch.float32,2),conic=g(field['conic'],torch.float32,4),depth=g(field['depth'],torch.float32,1),rgb=g(field['rgb'],torch.float32,3),rmin=g(minoff,torch.int32,2),rmax=g(maxoff,torch.int32,2),radii=r[3])
def main():
    assert not (OUT/'execution_plan.json').exists();plan=json.loads((OUT/'input_plan.json').read_text());assert sha(ROOT/'TGS_REAL_WORK_PROTOCOL_V7.md')==plan['protocol_sha256'];assert sha(ROOT/'tgs_real_work_probe.cu')==plan['source_sha256']
    receipt=json.loads((V5/'oracle/build_receipt.json').read_text());mp=Path(receipt['module_file']);assert sha(mp)==plan['native_module_expected']==receipt['module_sha256']
    spec=importlib.util.spec_from_file_location(mp.stem,mp);native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    from benchmark_tgs_backward import Workload
    from run_tgs_real_events import camera_tensors
    work=Workload();w,h=plan['resolution'];work.camera_metadata=plan['cameras'];work.cameras=[camera_tensors(c,w,h) for c in plan['cameras']]
    model_before={k:thash(getattr(work,k)) for k in ['points','sigma','opacity','sh','counts','offsets']}
    (OUT/'execution_plan.json').write_text(json.dumps(dict(input_plan_sha256=sha(OUT/'input_plan.json'),script_sha256=sha(Path(__file__)),model_before=model_before,module_sha256=sha(mp),workload=work.metadata,training=False,timing=False),indent=2)+'\n')
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    from torch.utils.cpp_extension import load
    build=OUT/'build';build.mkdir();flags=['-O3','--use_fast_math','--ptxas-options=-v']
    mod=load(name='tgs_diag_real_work_v7',sources=[str(ROOT/'tgs_real_work_probe.cu')],extra_include_paths=[str(V5/'generated/native/cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=flags,build_directory=str(build),verbose=True)
    (OUT/'build_receipt.json').write_text(json.dumps(dict(module_file=mod.__file__,module_sha256=sha(Path(mod.__file__)),source_sha256=sha(ROOT/'tgs_real_work_probe.cu'),helper_sha256=sha(V5/'generated/native/cuda_rasterizer/stable_window.cuh'),flags=flags,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()),indent=2)+'\n')
    s=subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',mod.__file__],capture_output=True,text=True,timeout=60);(OUT/'sass.txt').write_text(s.stdout);assert s.returncode==0
    rows=[]
    for ci,cam in enumerate(plan['cameras']):
        r=work.forward(native,ci);torch.cuda.synchronize();v=views(r,work.n,w,h)
        before={k:thash(t) for k,t in v.items()};cpu={k:t.cpu().numpy().copy() for k,t in v.items() if k not in ['ids','ranges','rgb','conic','depth']}
        visible=cpu['radii']>0;idsvis=np.flatnonzero(visible);mn,mx=cpu['rmin'][visible],cpu['rmax'][visible];assert int(((mx-mn).prod(1).astype(np.int64)).sum())==int(r[0])
        assert torch.all((v['ids']>=0)&(v['ids']<work.n));assert torch.all((v['ranges']>=0)&(v['ranges']<=int(r[0])));assert torch.all(v['ranges'][:,1]>=v['ranges'][:,0]);assert int((v['ranges'][:,1]-v['ranges'][:,0]).sum())==int(r[0])
        assert np.isfinite(cpu['coeff'][visible]).all();assert np.all(cpu['anchor'][visible]==np.floor(cpu['anchor'][visible]))
        localmin=np.column_stack([16*mn[:,0],16*mn[:,1]])-cpu['anchor'][visible]
        localmax=np.minimum(16*mx,np.array([w,h]))-1-cpu['anchor'][visible];Bcpu=np.maximum(np.abs(localmin),np.abs(localmax))
        ranks=np.array([int.from_bytes(hashlib.sha256(('tgs_real_work_v7/cache/'+cam['img_name']+'/'+str(i)).encode()).digest()[:8],'big') for i in idsvis],dtype=np.uint64)
        auditids=np.unique(np.r_[idsvis[np.argsort(ranks,kind='stable')[:512]],idsvis[np.argsort(-(Bcpu[:,0]*Bcpu[:,1]),kind='stable')[:32]]]).astype(np.int64)
        smap=torch.full((w*h,),-1,dtype=torch.int32,device='cuda');samplepixels=torch.tensor(plan['sample_pixels'][ci],dtype=torch.long,device='cuda');smap[samplepixels]=torch.arange(512,dtype=torch.int32,device='cuda')
        (OUT/f'camera_{ci}_input.json').write_text(json.dumps(dict(camera=cam,input_hashes=before,audit_triangles=auditids.tolist(),sorted_list_sha256=before['ids'],ranges_sha256=before['ranges']),indent=2)+'\n')
        q=mod.profile(v['ranges'],v['ids'],v['coeff'],v['anchor'],v['conic'],work.sigma,v['depth'],v['rgb'],v['rmin'],v['rmax'],v['radii'],smap,w,h,512);torch.cuda.synchronize()
        names=['counts','rgb','aux','path','maxblend','samples','split','cache','B','domain'];data={k:t.cpu().numpy() for k,t in zip(names,q)}
        pathoff=align(12*w*h);originalpath=r[6].narrow(0,pathoff,8*w*h).view(torch.int32).reshape(2,h,w).cpu().numpy()
        eq=dict(rgb=np.array_equal(data['rgb'],r[1].cpu().numpy()),aux=np.array_equal(data['aux'],r[2].cpu().numpy()),path=np.array_equal(data['path'],originalpath),maxblend=np.array_equal(data['maxblend'],r[9].cpu().numpy()))
        old={}
        if ci<4:
            with np.load(V5/f'real_{ci}_oracle.npz') as z:old={k:bool(np.array_equal(t,z[key])) for k,key,t in [('rgb','rgb',data['rgb']),('aux','aux',data['aux']),('path','contributors',data['path'].view(np.uint32)),('maxblend','maxblend',data['maxblend']),('radii','radii',cpu['radii']),('coeff','double_coeff',np.where(visible[:,None],cpu['coeff'],0))]}
        after={k:thash(t) for k,t in v.items()};assert before==after
        ct=data['counts'].astype(np.int64);total=ct.sum(0);identity=bool(np.all(ct[:,0]==ct[:,1:7].sum(1)) and np.all(ct[:,0]==ct[:,7]+ct[:,8]+ct[:,9]+ct[:,10]) and np.all(ct[:,10]<=1) and np.all(ct[:,16]+ct[:,17]==ct[:,3]) and np.all(ct[:,22]+ct[:,23]==ct[:,16]))
        # Save originals and new observation even if parity/gates fail; fail only AFTER preservation.
        np.savez_compressed(OUT/f'camera_{ci}_raw.npz',counts=data['counts'],rgb=data['rgb'],aux=data['aux'],path=data['path'],maxblend=data['maxblend'],samples=data['samples'],domain=data['domain'],B=data['B'],audit_triangles=auditids,audit_split=data['split'][auditids],audit_cache=data['cache'][auditids],coeff=np.where(visible[:,None],cpu['coeff'],0),anchors=np.where(visible[:,None],cpu['anchor'],0),rmin=np.where(visible[:,None],cpu['rmin'],0),rmax=np.where(visible[:,None],cpu['rmax'],0),radii=cpu['radii'],native_rgb=r[1].cpu().numpy(),native_aux=r[2].cpu().numpy(),native_path=originalpath,native_maxblend=r[9].cpu().numpy())
        K=ct[:,9];hist=np.bincount(K);np.save(OUT/f'camera_{ci}_K_histogram.npy',hist)
        row=dict(camera=ci,name=cam['img_name'],split=cam['split'],new_holdout=bool(cam.get('heldout_for_v7',False)),pixels=w*h,native_tiles=int(r[0]),visible_triangles=int(visible.sum()),domain_triangles=int(data['domain'][visible].sum()),parity=eq,old_v5_parity=old,input_hashes_preserved=before==after,partition_identity=identity,totals=total.tolist(),K_quantiles=np.quantile(K,[0,.5,.9,.95,.99,1]).tolist(),K_overflow={str(k):int((K>k).sum()) for k in [16,32,64,128,256]},sample_pairs=int((data['samples'][:,0]>0).sum()),diagnostic_buffer_hashes={k:thash(t) for k,t in zip(names,q) if k in ['split','cache','B','domain']},gate=bool(all(eq.values()) and all(old.values()) and identity and total[11:16].sum()==0))
        rows.append(row);(OUT/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
        del r,v,q,data,cpu,ct,smap,samplepixels
    after={k:thash(getattr(work,k)) for k in model_before};assert after==model_before
    result=dict(rows=rows,all_gates=all(r['gate'] for r in rows),model_hashes_preserved=True,input_plan_sha256=sha(OUT/'input_plan.json'),training=False,timing=False,renderer_candidate=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_gates'];print(json.dumps(dict(all_gates=True,cameras=len(rows),training=False)),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'execution_failure.txt').write_text(traceback.format_exc());raise
