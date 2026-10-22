"""Small coefficient/initialization probe; copied expression context is explicit."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch
from research.exact_perspective_backend import ROOT,RASTER,FLAGS

def payloads():
    helper=(ROOT/'research/exact_perspective_ray.inc').read_text(encoding='utf-8')
    cuda='''
#include "config.h"
#include <cuda_runtime.h>
'''+helper+'''
__global__ void coefficients(int n,const float2* xy,const float4* con,const float4* plane,
 const float* factors,int W,int H,float fx,float fy,float px,float py,float* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float2 q={(px-static_cast<float>(W-1)/2.f)/fx,(py-static_cast<float>(H-1)/2.f)/fy};
 float2 d={xy[i].x-px,xy[i].y-py};float4 c=con[i],p=plane[i];
 float power=-0.5f*(c.x*d.x*d.x+c.z*d.y*d.y)-c.y*d.x*d.y;
 float alpha=fminf(.99f,c.w*expf(power));
 float* o=out+13*i;o[0]=power;o[1]=alpha;o[2]=(!(power>0.f)&&!(alpha<1.f/255.f))?1.f:0.f;
 o[3]=p.x*d.x+p.y*d.y+p.z;o[4]=p.w;
 auto a=exact_packet_ray(xy[i],p,c,q,W,H,fx,fy);
 auto b=exact_factor_ray(factors+12*i,c,q);
 o[5]=a.peak;o[6]=a.reciprocal_sigma;o[7]=a.alpha;o[8]=a.valid?1.f:0.f;
 o[9]=b.peak;o[10]=b.reciprocal_sigma;o[11]=b.alpha;o[12]=b.valid?1.f:0.f;
}
__global__ void replay_initialization(int n,const float* coeff,int W,int H,float fx,float fy,float px,float py,float* out){
 if(blockIdx.x||threadIdx.x)return;
 float T=1.f,init=0.f;int last=0;
 for(int i=0;i<n;i++){
   const float* c=coeff+13*i;if(c[2]==0)continue;
   float alpha=c[1],test=T*(1.f-alpha);if(test<.0001f)break;
   init=T>.5f?c[3]:init;T=test;last=i+1;
 }
 float qx=(px-static_cast<float>(W-1)/2.f)/fx,qy=(py-static_cast<float>(H-1)/2.f)/fy;
 out[0]=init;out[1]=fmaxf(init-SAMPLE_RANGE,0.f);out[2]=fmaxf(init+SAMPLE_RANGE,0.f);
 out[3]=T;out[4]=float(last);out[5]=rnorm3df(qx,qy,1.f);out[6]=qx;out[7]=qy;
 out[8]=T<=MIN_TRANSMITTANCE?1.f:0.f;
}
void launch(int n,const float* xy,const float* c,const float* p,const float* f,int W,int H,
 float fx,float fy,float px,float py,float* out,float* init,cudaStream_t stream){
 coefficients<<<(n+127)/128,128,0,stream>>>(n,(const float2*)xy,(const float4*)c,(const float4*)p,f,W,H,fx,fy,px,py,out);
 replay_initialization<<<1,1,0,stream>>>(n,out,W,H,fx,fy,px,py,init);
}
'''
    cpp='''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void launch(int,const float*,const float*,const float*,const float*,int,int,float,float,float,float,float*,float*,cudaStream_t);
std::vector<torch::Tensor> forward(torch::Tensor xy,torch::Tensor c,torch::Tensor p,torch::Tensor f,
 int W,int H,double fx,double fy,double px,double py){
 for(auto t:{xy,c,p,f})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==torch::kFloat32,"CUDA contiguous FP32 required");
 c10::cuda::CUDAGuard guard(xy.device());auto n=xy.size(0);
 TORCH_CHECK(xy.size(1)==2&&c.size(0)==n&&c.size(1)==4&&p.sizes()==c.sizes()&&f.size(0)==n&&f.size(1)==12,"Bad shapes");
 auto out=torch::empty({n,13},xy.options()),init=torch::empty({9},xy.options());
 launch(n,xy.data_ptr<float>(),c.data_ptr<float>(),p.data_ptr<float>(),f.data_ptr<float>(),W,H,fx,fy,px,py,out.data_ptr<float>(),init.data_ptr<float>(),at::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,init};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
'''
    return {'capture.cu':cuda,'binding.cpp':cpp,'config.h':(RASTER/'cuda_rasterizer/config.h').read_text(encoding='utf-8')}

def load(out):
    data=payloads();h=hashlib.sha256(torch.__version__.encode())
    for name,text in sorted(data.items()):h.update(name.encode());h.update(text.encode())
    signature=h.hexdigest()[:12];build=out/'generated';build.mkdir()
    for name,text in data.items():(build/name).write_text(text,encoding='utf-8')
    from torch.utils.cpp_extension import load
    module=load(name=f'exact_outlier_capture_{signature}',sources=[str(build/n) for n in ('binding.cpp','capture.cu')],extra_cflags=['-O3'],
        extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
    return module,{'signature':signature,'binary_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        'generated_sha256':{n:hashlib.sha256((build/n).read_bytes()).hexdigest() for n in data},'flags':FLAGS,
        'scope':'Copied exact helper and affine eligibility/initialization expressions; not original kernel instruction trace.'}
