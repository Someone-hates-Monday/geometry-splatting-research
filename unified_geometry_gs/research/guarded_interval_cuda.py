"""Device endpoint diagnostic for shared strict intervals; not a GPU root solver."""
from pathlib import Path
import hashlib
import torch
FLAGS=['-O3','--fmad=false','--ftz=false','--prec-div=true','--prec-sqrt=true','--allow-unsupported-compiler']

def load(out):
    build=out/'generated';build.mkdir()
    cu=r'''
#include "guarded_interval.cuh"
__global__ void run(const double* layers,const int* sections,const double* endpoint,double* result,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int start=sections[2*i],n=sections[2*i+1];gi::Layer c[192];
 if(n>192){result[4*i+3]=0.;return;}
 for(int j=0;j<n;j++){const double* p=layers+7*(start+j);c[j]={{p[0],p[1]},{p[2],p[3]},{p[4],p[5]},p[6]};}
 double scale=0.;bool okay=false;gi::I r=gi::residual(c,n,endpoint[i],scale,okay);
 result[4*i]=r.lo;result[4*i+1]=r.hi;result[4*i+2]=scale;result[4*i+3]=okay?1.:0.;
}
void launch(const double* a,const int* b,const double* c,double* d,int n,cudaStream_t stream){run<<<(n+63)/64,64,0,stream>>>(a,b,c,d,n);}
'''
    cpp=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void launch(const double*,const int*,const double*,double*,int,cudaStream_t);
torch::Tensor forward(torch::Tensor a,torch::Tensor b,torch::Tensor c){
 TORCH_CHECK(a.is_cuda()&&b.is_cuda()&&c.is_cuda()&&a.is_contiguous()&&b.is_contiguous()&&c.is_contiguous(),"CUDA contiguous required");
 TORCH_CHECK(a.scalar_type()==torch::kFloat64&&a.dim()==2&&a.size(1)==7&&b.scalar_type()==torch::kInt32&&b.dim()==2&&b.size(1)==2&&c.scalar_type()==torch::kFloat64&&c.dim()==1&&b.size(0)==c.size(0),"Shapes/dtypes invalid");
 TORCH_CHECK(a.device()==b.device()&&a.device()==c.device(),"All tensors must share a device");
 c10::cuda::CUDAGuard guard(a.device());auto out=torch::full({c.size(0),4},NAN,a.options());
 launch(a.data_ptr<double>(),b.data_ptr<int>(),c.data_ptr<double>(),out.data_ptr<double>(),c.size(0),at::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
'''
    files={'core.cu':cu,'binding.cpp':cpp,'guarded_interval.cuh':Path(__file__).with_name('guarded_interval.cuh').read_text(encoding='utf-8')}
    h=hashlib.sha256(torch.__version__.encode())
    for name,source in sorted(files.items()):h.update(name.encode());h.update(source.encode());(build/name).write_text(source,encoding='utf-8')
    from torch.utils.cpp_extension import load
    return load(name='guarded_interval_'+h.hexdigest()[:12],sources=[str(build/'binding.cpp'),str(build/'core.cu')],extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
