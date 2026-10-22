"""Actual query capture + strict streaming wrapper of the frozen cache decoder."""
from pathlib import Path
import hashlib
def once(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
def query_payloads(base):
    data=dict(base)
    data['rasterize_points.cu']=once(data['rasterize_points.cu'],'torch::full({6, H, W}, 0.0, float_opts)','torch::full({8, H, W}, 0.0, float_opts)')
    data['cuda_rasterizer/render_forward.cu']=once(data['cuda_rasterizer/render_forward.cu'],
        'out_mdepth[5 * H * W + pix_id] = T<=MIN_TRANSMITTANCE?1.f:0.f;',
        'out_mdepth[5 * H * W + pix_id] = T<=MIN_TRANSMITTANCE?1.f:0.f;\n'
        '            out_mdepth[6 * H * W + pix_id] = pixnf.x;\n'
        '            out_mdepth[7 * H * W + pix_id] = pixnf.y;')
    return data
def compile_module(out,folder,files,flags,name,include=()):
    import torch
    from torch.utils.cpp_extension import load
    b=out/folder;b.mkdir();h=hashlib.sha256((torch.__version__+str(flags)).encode())
    for n,s in sorted(files.items()):h.update(n.encode());h.update(s.encode());p=b/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
    sources=[str(b/n) for n in files if n.endswith(('.cu','.cpp'))]
    m=load(name=name+h.hexdigest()[:12],sources=sources,extra_include_paths=[str(b),str(b/'cuda_rasterizer'),*include],extra_cflags=['-O3'],extra_cuda_cflags=flags,build_directory=str(b),verbose=True)
    p=Path(m.__file__)
    return m,dict(binary_path=str(p),binary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),generated_sha256={n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in files},flags=flags)
def load_query(out,old):
    from research.exact_perspective_backend import FLAGS,RASTER
    p=old/'instrumented_generated/src';base={x.relative_to(p).as_posix():x.read_text(encoding='utf-8') for x in p.rglob('*') if x.suffix in ('.cu','.cpp','.h')}
    return compile_module(out,'query_generated',query_payloads(base),FLAGS,'actual_query_',include=[str(RASTER),str(RASTER/'cuda_rasterizer'),str(RASTER/'third_party/glm')])
def load_stream(out,old):
    from research.guarded_interval_cuda import FLAGS
    files={n:(old/'generated'/n).read_text(encoding='utf-8') for n in ['guarded_interval.cuh','guarded_preprocess.cuh','shared_cache.cuh']}
    files['stream.cu']=r'''
#include "shared_cache.cuh"
__global__ void stream_kernel(const double* cache,const int* records,const int64_t* offsets,const float* query,const float* amp,double* output,int64_t start,int count,int P,int pixels){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;int64_t index=start+i;int id=records[2*index];int lo=0,hi=pixels+1;
 while(lo<hi){int mid=lo+(hi-lo)/2;if(offsets[mid]<=index)lo=mid+1;else hi=mid;}int pixel=lo-1;
 if(id<0||id>=P||pixel<0||pixel>=pixels||index>=offsets[pixel+1]){output[52*i]=output[52*i+26]=6;return;}
 double src[36];src[32]=double(query[2*pixel]);src[33]=double(query[2*pixel+1]);src[34]=1.;src[35]=double(amp[id]);
 for(int mode=0;mode<2;mode++)gc::decode_one(src,cache+gc::CACHE_WIDTH*id,mode,output+(2*i+mode)*26);
}
void stream_launch(const double* a,const int* b,const int64_t* c,const float* d,const float* e,double* f,int64_t start,int n,int P,int pixels,cudaStream_t s){stream_kernel<<<(n+63)/64,64,0,s>>>(a,b,c,d,e,f,start,n,P,pixels);}
'''
    files['binding.cpp']=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void stream_launch(const double*,const int*,const int64_t*,const float*,const float*,double*,int64_t,int,int,int,cudaStream_t);
torch::Tensor stream(torch::Tensor cache,torch::Tensor records,torch::Tensor offsets,torch::Tensor query,torch::Tensor amp,int64_t start,int count){
 for(auto t:{cache,records,offsets,query,amp})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==cache.device(),"Same-device contiguous tensors required");
 TORCH_CHECK(cache.scalar_type()==torch::kFloat64&&cache.dim()==2&&cache.size(1)==28&&records.scalar_type()==torch::kInt32&&records.dim()==2&&records.size(1)==2,"Cache/records invalid");
 TORCH_CHECK(offsets.scalar_type()==torch::kInt64&&offsets.dim()==1&&query.scalar_type()==torch::kFloat32&&query.dim()==2&&query.size(1)==2&&offsets.numel()==query.size(0)+1&&amp.scalar_type()==torch::kFloat32&&amp.dim()==1&&amp.size(0)==cache.size(0),"Query/AMP/offsets invalid");
 TORCH_CHECK(start>=0&&count>=0&&count<=65536&&start+count<=records.size(0),"Stream bounds invalid");
 c10::cuda::CUDAGuard guard(cache.device());auto out=torch::full({count,2,26},NAN,cache.options());
 stream_launch(cache.data_ptr<double>(),records.data_ptr<int>(),offsets.data_ptr<int64_t>(),query.data_ptr<float>(),amp.data_ptr<float>(),out.data_ptr<double>(),start,count,cache.size(0),query.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("stream",&stream);}
'''
    return compile_module(out,'stream_generated',files,FLAGS,'csr_decode_')
