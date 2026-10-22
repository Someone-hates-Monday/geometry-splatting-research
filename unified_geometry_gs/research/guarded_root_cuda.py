"""Source-parameter GPU preprocessing and independently certified device root candidates."""
from pathlib import Path
import hashlib
import torch
from research.guarded_interval_cuda import FLAGS

def load(out):
    build=out/'generated';build.mkdir()
    cu=r'''
#include "guarded_root.cuh"
void preprocess(const double* a,double* b,int n,cudaStream_t stream){gp::preprocess<<<(n+63)/64,64,0,stream>>>(n,a,b);}
void solve(const double* a,const int* b,double* c,int n,cudaStream_t stream){gr::solve<<<(n+31)/32,32,0,stream>>>(n,a,b,c);}
'''
    cpp=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void preprocess(const double*,double*,int,cudaStream_t);void solve(const double*,const int*,double*,int,cudaStream_t);
torch::Tensor pre(torch::Tensor a){TORCH_CHECK(a.is_cuda()&&a.is_contiguous()&&a.scalar_type()==torch::kFloat64&&a.dim()==2&&a.size(1)==36,"Source CUDA double rows36 required");c10::cuda::CUDAGuard guard(a.device());auto out=torch::full({a.size(0),2,26},NAN,a.options());preprocess(a.data_ptr<double>(),out.data_ptr<double>(),a.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
torch::Tensor run(torch::Tensor a,torch::Tensor b){TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&a.device()==b.device()&&a.is_contiguous()&&b.is_contiguous()&&a.scalar_type()==torch::kFloat64&&a.dim()==3&&a.size(1)==2&&a.size(2)==26&&b.scalar_type()==torch::kInt32&&b.dim()==2&&b.size(1)==2,"Same-device packets and sections required");c10::cuda::CUDAGuard guard(a.device());auto out=torch::full({b.size(0),12},NAN,a.options());solve(a.data_ptr<double>(),b.data_ptr<int>(),out.data_ptr<double>(),b.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("preprocess",&pre);m.def("solve",&run);}
'''
    sources={'root.cu':cu,'binding.cpp':cpp}
    for n in ('guarded_interval.cuh','guarded_preprocess.cuh','guarded_root.cuh'):sources[n]=Path(__file__).with_name(n).read_text(encoding='utf-8')
    h=hashlib.sha256(torch.__version__.encode())
    for n,t in sorted(sources.items()):h.update(n.encode());h.update(t.encode());(build/n).write_text(t,encoding='utf-8')
    from torch.utils.cpp_extension import load
    return load(name='guarded_root_'+h.hexdigest()[:12],sources=[str(build/'binding.cpp'),str(build/'root.cu')],extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
