from pathlib import Path
from research.csr_decode_cuda import compile_module
from research.float_product_cuda import recip_table
def load(out):
    r=Path(__file__).parent;old=r/'experiments/short_certificate_v1/variant_generated';cache=r/'experiments/shared_cache_v1/generated'
    files={n:(old/n).read_text(encoding='utf-8') for n in ['guarded_interval.cuh','short_interval.cuh']}
    files.update({n:(cache/n).read_text(encoding='utf-8') for n in ['guarded_preprocess.cuh','shared_cache.cuh']})
    files.update({n:(r/n).read_text(encoding='utf-8') for n in ['float_product.cuh','dynamic_root.cuh']});files['float_reciprocals.cuh']=recip_table()
    files['root.cu']=r'''
#include "dynamic_root.cuh"
void decode_launch(const double* c,const int* ids,const int64_t* s,const double* q,const float* a,double* p,int64_t n,int m,int mode,int P,cudaStream_t st){dr::decode<<<(n+63)/64,64,0,st>>>(c,ids,s,q,a,p,n,m,mode,P);}
void root_launch(const double* p,const int64_t* s,const double* q,double* o,int n,int mode,cudaStream_t st){dr::solve<<<(n*32+127)/128,128,0,st>>>(p,s,q,o,n,mode);}
'''
    files['binding.cpp']=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void decode_launch(const double*,const int*,const int64_t*,const double*,const float*,double*,int64_t,int,int,int,cudaStream_t);
void root_launch(const double*,const int64_t*,const double*,double*,int,int,cudaStream_t);
torch::Tensor decode(torch::Tensor c,torch::Tensor ids,torch::Tensor s,torch::Tensor q,torch::Tensor a,int mode){for(auto t:{c,ids,s,q,a})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==c.device(),"decode device");TORCH_CHECK(c.scalar_type()==torch::kFloat64&&c.dim()==2&&c.size(1)==28&&ids.scalar_type()==torch::kInt32&&ids.dim()==1&&s.scalar_type()==torch::kInt64&&s.dim()==1&&q.scalar_type()==torch::kFloat64&&q.dim()==2&&q.size(1)==4&&s.numel()==q.size(0)+1&&a.scalar_type()==torch::kFloat32&&a.dim()==1&&a.numel()==c.size(0)&&mode>=0&&mode<=1,"decode shape");TORCH_CHECK(s[0].item<int64_t>()==0&&s[-1].item<int64_t>()==ids.numel(),"decode sections");TORCH_CHECK(ids.numel()>0&&ids.min().item<int>()>=0&&ids.max().item<int>()<c.size(0),"decode ids");c10::cuda::CUDAGuard g(c.device());auto p=torch::full({ids.numel(),26},NAN,c.options());decode_launch(c.data_ptr<double>(),ids.data_ptr<int>(),s.data_ptr<int64_t>(),q.data_ptr<double>(),a.data_ptr<float>(),p.data_ptr<double>(),ids.numel(),q.size(0),mode,c.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return p;}
torch::Tensor solve(torch::Tensor p,torch::Tensor s,torch::Tensor q,int mode){for(auto t:{p,s,q})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.device()==p.device(),"root device");TORCH_CHECK(p.scalar_type()==torch::kFloat64&&p.dim()==2&&p.size(1)==26&&s.scalar_type()==torch::kInt64&&s.dim()==1&&q.scalar_type()==torch::kFloat64&&q.dim()==2&&q.size(1)==4&&s.numel()==q.size(0)+1&&mode>=0&&mode<=1,"root shape");TORCH_CHECK(s[0].item<int64_t>()==0&&s[-1].item<int64_t>()==p.size(0),"root sections");c10::cuda::CUDAGuard g(p.device());auto o=torch::full({q.size(0),20},NAN,p.options());root_launch(p.data_ptr<double>(),s.data_ptr<int64_t>(),q.data_ptr<double>(),o.data_ptr<double>(),q.size(0),mode,at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return o;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode",&decode);m.def("solve",&solve);}
'''
    from research.guarded_interval_cuda import FLAGS
    return compile_module(out,'generated',files,FLAGS,'dynamic_root_')
