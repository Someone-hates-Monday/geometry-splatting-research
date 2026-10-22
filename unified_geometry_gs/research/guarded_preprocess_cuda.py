"""Device source-parameter budgets, mass and saved-bracket endpoint diagnostics."""
from pathlib import Path
import hashlib
import torch
from research.guarded_interval_cuda import FLAGS

def load(out):
    build=out/'generated';build.mkdir()
    cu=r'''
#include "guarded_preprocess.cuh"
void preprocess(const double* a,double* b,int n,cudaStream_t stream){gp::preprocess<<<(n+63)/64,64,0,stream>>>(n,a,b);}
void mass(const double* a,const int* b,double* c,int n,cudaStream_t stream){gp::mass<<<(2*n+63)/64,64,0,stream>>>(n,a,b,c);}
void endpoints(const double* a,const int* b,const double* c,double* d,int n,cudaStream_t stream){gp::endpoints<<<(2*n+63)/64,64,0,stream>>>(n,a,b,c,d);}
'''
    cpp=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void preprocess(const double*,double*,int,cudaStream_t);void mass(const double*,const int*,double*,int,cudaStream_t);
void endpoints(const double*,const int*,const double*,double*,int,cudaStream_t);
void floating(torch::Tensor a){TORCH_CHECK(a.is_cuda()&&a.is_contiguous()&&a.scalar_type()==torch::kFloat64,"CUDA contiguous double required");}
void sections(torch::Tensor a,torch::Tensor b){TORCH_CHECK(b.is_cuda()&&b.is_contiguous()&&b.scalar_type()==torch::kInt32&&b.dim()==2&&b.size(1)==2&&a.device()==b.device(),"CUDA same-device int32 sections required");}
torch::Tensor pre(torch::Tensor a){floating(a);TORCH_CHECK(a.dim()==2&&a.size(1)==36,"Source rows require36");c10::cuda::CUDAGuard guard(a.device());auto out=torch::full({a.size(0),2,26},NAN,a.options());preprocess(a.data_ptr<double>(),out.data_ptr<double>(),a.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
torch::Tensor classify(torch::Tensor a,torch::Tensor b){floating(a);sections(a,b);TORCH_CHECK(a.dim()==3&&a.size(1)==2&&a.size(2)==26,"Packets invalid");c10::cuda::CUDAGuard guard(a.device());auto out=torch::full({b.size(0),2,3},NAN,a.options());mass(a.data_ptr<double>(),b.data_ptr<int>(),out.data_ptr<double>(),b.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
torch::Tensor check(torch::Tensor a,torch::Tensor b,torch::Tensor c){floating(a);floating(c);sections(a,b);TORCH_CHECK(a.dim()==3&&a.size(1)==2&&a.size(2)==26&&c.device()==a.device()&&c.dim()==1&&c.size(0)==b.size(0),"Endpoint input invalid");c10::cuda::CUDAGuard guard(a.device());auto out=torch::full({b.size(0),2,4},NAN,a.options());endpoints(a.data_ptr<double>(),b.data_ptr<int>(),c.data_ptr<double>(),out.data_ptr<double>(),b.size(0),at::cuda::getCurrentCUDAStream());C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("preprocess",&pre);m.def("classify",&classify);m.def("check",&check);}
'''
    sources={'preprocess.cu':cu,'binding.cpp':cpp}
    for name in ('guarded_interval.cuh','guarded_preprocess.cuh'):sources[name]=Path(__file__).with_name(name).read_text(encoding='utf-8')
    h=hashlib.sha256(torch.__version__.encode())
    for name,text in sorted(sources.items()):h.update(name.encode());h.update(text.encode());(build/name).write_text(text,encoding='utf-8')
    from torch.utils.cpp_extension import load
    return load(name='guarded_preprocess_'+h.hexdigest()[:12],sources=[str(build/'binding.cpp'),str(build/'preprocess.cu')],extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
