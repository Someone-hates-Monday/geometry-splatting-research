"""Bounded libdevice dispatch experiment, not a renderer fix or accuracy proof."""
import hashlib,json,os,subprocess,sys,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/tgs_standard_pow_dispatch_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
SOURCE=r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
// Low-level libdevice ABI experiment; no assumption of a supplied CUDA C++ header declaration.
extern "C" __device__ float __nv_powf(float,float);
__global__ void kernel(const float* in,float* out,int n){int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n)return;float u=in[2*j],s=in[2*j+1];out[3*j]=__powf(u,s);out[3*j+1]=__nv_powf(u,s);out[3*j+2]=powf(u,s);}
torch::Tensor probe(torch::Tensor in){TORCH_CHECK(in.is_cuda()&&in.is_contiguous()&&in.scalar_type()==torch::kFloat32&&in.dim()==2&&in.size(1)==2);auto out=torch::zeros({in.size(0),3},in.options());kernel<<<(in.size(0)+127)/128,128>>>(in.data_ptr<float>(),out.data_ptr<float>(),in.size(0));TORCH_CHECK(cudaGetLastError()==cudaSuccess);return out;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("probe",&probe);}
'''
def main():
    assert not OUT.exists();OUT.mkdir();v6=ROOT/'experiments/tgs_compensated_edge_v6';trace=ROOT/'experiments/tgs_compensated_renderer_v8/depth_trace';ta=json.loads((trace/'independent_audit.json').read_text());assert ta['all_native_endpoints_bitwise']
    with np.load(v6/'inputs.npz') as z:inputs=[z['power'].copy()]
    n0=len(inputs[0]);raw=(v6/'strict/power.bin').read_bytes();ref=[np.frombuffer(raw[:n0*8],'<f4').reshape(n0,2)[:,0].copy()];labels=[['v6',int(i),-1] for i in range(n0)];sourcehash={str((v6/p).relative_to(ROOT)):sha(v6/p) for p in ['inputs.npz','strict/power.bin','evidence_manifest.json']}
    for pi in range(14):
        pp=trace/f'packet_{pi}.npz';tp=trace/f'pixel_{pi}_guard.npz';sourcehash[str(pp.relative_to(ROOT))]=sha(pp);sourcehash[str(tp.relative_to(ROOT))]=sha(tp)
        with np.load(pp) as z:packed=z['packed']
        with np.load(tp) as z:t=z['trace'];t=t[np.isfinite(t[:,0])]
        ords=t[:,1].astype(int)-1;assert np.array_equal(packed[ords,56],t[:,0]);inputs.append(np.c_[t[:,30].astype(np.float32),packed[ords,2]].astype(np.float32));ref.append(t[:,33].astype(np.float32));labels.extend([['trace',pi,int(j+1)] for j in ords])
    xy=np.concatenate(inputs).astype(np.float32);reference=np.concatenate(ref).astype(np.float32);np.savez_compressed(OUT/'input.npz',power=xy,standard_reference=reference);(OUT/'labels.json').write_text(json.dumps(labels)+'\n');build=OUT/'build';build.mkdir();src=build/'probe.cu';src.write_text(SOURCE)
    plan=dict(script_sha256=sha(Path(__file__)),source_sha256=sha(src),input_sha256=sha(OUT/'input.npz'),source_hashes=sourcehash,v6_rows=n0,total_rows=len(xy),flags=['-O3','--use_fast_math','--ftz=true'],prototype='Explicit experimental extern C device declaration of documented LLVM/libdevice __nv_powf. No claim of a public header API.',primary_source='https://docs.nvidia.com/cuda/archive/12.8.0/libdevice-users-guide/__nv_powf.html',primary_query='site.docs.nvidia.com cuda libdevice __nv_powf float fast_powf',training=False,timing=False,renderer=False)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    import torch
    from torch.utils.cpp_extension import load
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2');os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    m=load(name='tgs_diag_standard_pow_dispatch_v1',sources=[str(src)],extra_cflags=['-O3'],extra_cuda_cflags=plan['flags'],build_directory=str(build),verbose=True)
    value=m.probe(torch.from_numpy(xy).cuda());torch.cuda.synchronize();a=value.cpu().numpy();np.savez_compressed(OUT/'outputs.npz',values=a)
    sas=subprocess.run(['/usr/local/cuda-12.8/bin/cuobjdump','--dump-sass',m.__file__],capture_output=True,text=True,timeout=60);assert sas.returncode==0;(OUT/'sass.txt').write_text(sas.stdout)
    subsets={'all':np.ones(len(xy),bool),'normal_input_and_normal_output':(xy[:,0]>=2.**-126)&(reference>=2.**-126),'trace_rows':np.arange(len(xy))>=n0,'v6_subnormal_input':(np.arange(len(xy))<n0)&(xy[:,0]<2.**-126)};rows={}
    for key,mask in subsets.items():
        rows[key]=dict(rows=int(mask.sum()),explicit_vs_standard_reference_bitwise=int(np.count_nonzero(a[mask,1].view(np.uint32)==reference[mask].view(np.uint32))),explicit_abs_error=float(np.abs(a[mask,1].astype(float)-reference[mask].astype(float)).max()),explicit_vs_fast_different=int(np.count_nonzero(a[mask,1].view(np.uint32)!=a[mask,0].view(np.uint32))))
    result=dict(subsets=rows,fast_equals_named_bitwise=bool(np.array_equal(a[:,0],a[:,2])),explicit_distinct_from_fast=bool(np.any(a[:,0]!=a[:,1])),module_file=m.__file__,module_sha256=sha(Path(m.__file__)),SASS_sha256=sha(OUT/'sass.txt'),plan_sha256=sha(OUT/'plan.json'),torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),power_interval_proved=False,renderer_fix_proved=False,training=False,timing=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (OUT/'failure.txt').write_text(traceback.format_exc());raise
