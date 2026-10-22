"""One-line new observer version; preserve initial failure and all frozen inputs."""
import hashlib,importlib.util,json,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch
from profile_tgs_real_work import ROOT,OUT,V5,sha,thash,align,views
PATCH=OUT/'sentinel_v7_1'
def main():
    assert not PATCH.exists();PATCH.mkdir();plan=json.loads((OUT/'input_plan.json').read_text());initial=json.loads((OUT/'results.json').read_text());assert not initial['all_gates']
    source=ROOT/'tgs_real_work_probe.cu';assert sha(source)==plan['source_sha256'];text=source.read_text();assert text.count('mid=0xffffffff;')==1
    generated=PATCH/'tgs_real_work_probe_v7_1.cu';generated.write_text(text.replace('mid=0xffffffff;','mid=__float2uint_rz(-1.0f);'))
    correction=dict(addendum_sha256=sha(ROOT/'TGS_REAL_WORK_ADDENDUM_V7_1.md'),original_source_sha256=sha(source),source_sha256=sha(generated),script_sha256=sha(Path(__file__)),input_plan_sha256=sha(OUT/'input_plan.json'),initial_results_sha256=sha(OUT/'results.json'),initial_failure_sha256=sha(OUT/'execution_failure.txt'),training=False,timing=False)
    (PATCH/'plan.json').write_text(json.dumps(correction,indent=2)+'\n')
    receipt=json.loads((V5/'oracle/build_receipt.json').read_text());p=Path(receipt['module_file']);assert sha(p)==plan['native_module_expected'];s=importlib.util.spec_from_file_location(p.stem,p);native=importlib.util.module_from_spec(s);s.loader.exec_module(native)
    from benchmark_tgs_backward import Workload
    from run_tgs_real_events import camera_tensors
    work=Workload();w,h=plan['resolution'];work.cameras=[camera_tensors(c,w,h) for c in plan['cameras']]
    model_hashes={k:thash(getattr(work,k)) for k in ['points','sigma','opacity','sh','counts','offsets']};assert model_hashes==json.loads((OUT/'execution_plan.json').read_text())['model_before']
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    from torch.utils.cpp_extension import load
    build=PATCH/'build';build.mkdir();mod=load(name='tgs_diag_real_work_v7_1',sources=[str(generated)],extra_include_paths=[str(V5/'generated/native/cuda_rasterizer')],extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--ptxas-options=-v'],build_directory=str(build),verbose=True)
    (PATCH/'build_receipt.json').write_text(json.dumps(dict(module_file=mod.__file__,module_sha256=sha(Path(mod.__file__)),source_sha256=sha(generated)),indent=2)+'\n')
    sass=subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',mod.__file__],capture_output=True,text=True,timeout=60);(PATCH/'sass.txt').write_text(sass.stdout);assert sass.returncode==0
    rows=[];names=['counts','rgb','aux','path','maxblend','samples','split','cache','B','domain']
    for ci,cam in enumerate(plan['cameras']):
        r=work.forward(native,ci);torch.cuda.synchronize();v=views(r,work.n,w,h);before={k:thash(t) for k,t in v.items()};assert before==json.loads((OUT/f'camera_{ci}_input.json').read_text())['input_hashes']
        smap=torch.full((w*h,),-1,dtype=torch.int32,device='cuda');spix=torch.tensor(plan['sample_pixels'][ci],dtype=torch.long,device='cuda');smap[spix]=torch.arange(512,dtype=torch.int32,device='cuda')
        out=mod.profile(v['ranges'],v['ids'],v['coeff'],v['anchor'],v['conic'],work.sigma,v['depth'],v['rgb'],v['rmin'],v['rmax'],v['radii'],smap,w,h,512);torch.cuda.synchronize();data={k:t.cpu().numpy() for k,t in zip(names,out)}
        path=r[6].narrow(0,align(12*w*h),8*w*h).view(torch.int32).reshape(2,h,w).cpu().numpy();parity={k:bool(np.array_equal(data[k],t)) for k,t in [('rgb',r[1].cpu().numpy()),('aux',r[2].cpu().numpy()),('path',path),('maxblend',r[9].cpu().numpy())]}
        original=initial['rows'][ci];frozen={}
        with np.load(OUT/f'camera_{ci}_raw.npz') as z:
            for k in ['counts','rgb','aux','maxblend','samples','B','domain']:frozen[k]=bool(np.array_equal(data[k],z[k],equal_nan=True))
            frozen['split']=bool(np.array_equal(data['split'][z['audit_triangles']],z['audit_split']));frozen['cache']=bool(np.array_equal(data['cache'][z['audit_triangles']],z['audit_cache']))
            oldpath=z['path'];expected=oldpath.copy();expected[1][z['counts'][:,9].reshape(h,w)==0]=0
            frozen['only_declared_path_delta']=bool(np.array_equal(data['path'],expected));delta=int((data['path']!=oldpath).sum())
        dh={k:thash(t) for k,t in zip(names,out) if k in ['split','cache','B','domain']};assert dh==original['diagnostic_buffer_hashes'];assert before=={k:thash(t) for k,t in v.items()}
        np.savez_compressed(PATCH/f'camera_{ci}_path.npz',path=data['path'],native_path=path)
        row=dict(camera=ci,name=cam['img_name'],parity=parity,all_other_outputs_frozen=frozen,path_changed_elements=delta,original_input_hashes_identical=True,diagnostic_buffer_hashes=dh,gate=all(parity.values()) and all(frozen.values()));rows.append(row)
        (PATCH/'progress.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True);del r,v,out,data,smap,spix
    assert model_hashes=={k:thash(getattr(work,k)) for k in model_hashes}
    result=dict(rows=rows,all_gates=all(r['gate'] for r in rows),plan_sha256=sha(PATCH/'plan.json'),model_hashes_preserved=True,training=False,timing=False)
    (PATCH/'results.json').write_text(json.dumps(result,indent=2)+'\n');assert result['all_gates']
if __name__=='__main__':
    try:main()
    except BaseException:
        (PATCH/'failure.txt').write_text(traceback.format_exc());raise
