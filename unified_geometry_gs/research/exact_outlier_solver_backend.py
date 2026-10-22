"""Controlled CUDA T-evaluation replacements on serialized FP32 coefficients."""
from pathlib import Path
import hashlib
import torch
FLAGS=['-O3','--use_fast_math','--allow-unsupported-compiler']

def payloads():
    cu=r'''
#include <cuda_runtime.h>
#include <cmath>
#include "config.h"
__global__ void solve(int n,const float* c,const float* init,double* output){
 int mode=threadIdx.x;if(blockIdx.x||mode>=3)return;
 double* o=output+65*mode;float lo=init[0],hi=init[1],rln=init[3];
 double tp[9];bool in_range=init[2]<=MIN_TRANSMITTANCE;bool first_valid=false;
 for(int it=0;it<5;it++){
   int start=it==0?0:1,end=it==0?9:8;float interval=(hi-lo)*.125f;
   for(int s=start;s<end;s++){
     float ts=lo+interval*s;float product32=1.f;double product64=1.,lt=0.;
     for(int i=0;i<n;i++){
       float p=c[3*i],rho=c[3*i+1],a=c[3*i+2];
       if(mode<2){
         float delta=(ts-p)*rho;float g=expf(-.5f*delta*delta);
         float one=1.f-a*g;float rv=rsqrtf(one);
         float factor=(ts>p?(1.f-a):one)*rv;
         if(mode==0)product32*=factor;else product64*=double(factor);
       }else{
         double delta=(double(ts)-double(p))*double(rho);
         double g=exp(-.5*delta*delta),lg=log1p(-double(a)*g);
         lt+=(ts>p?log1p(-double(a))-.5*lg:.5*lg);
       }
     }
     tp[s]=mode==0?double(product32):(mode==1?product64:exp(lt));
   }
   double* tr=o+10+11*it;tr[0]=lo;tr[1]=hi;
   for(int s=0;s<9;s++)tr[2+s]=tp[s];
   if(it==0){in_range=tp[0]>=.5&&tp[8]<=.5&&in_range;first_valid=in_range;}
   int id=0;for(int s=1;s<8;s++)id=tp[s]>=.5?s:id;
   hi=lo+(id+1)*interval;lo=lo+id*interval;
   double left=tp[id],right=tp[id+1];tp[0]=left;tp[8]=right;
 }
 float ratio=mode==0?(float(tp[0])-.5f)/(float(tp[0])-float(tp[8])):float((tp[0]-.5)/(tp[0]-tp[8]));
 float w=__saturatef(ratio),root=in_range?w*hi+(1.f-w)*lo:0.f;
 float slope=0.f;
 if(in_range){float denom=fmaxf(hi-lo,1e-6f);slope=mode==0?fabsf(float(tp[0])-float(tp[8]))/denom:float(fabs(tp[0]-tp[8])/double(denom));}
 o[0]=root;o[1]=root*rln;o[2]=slope;o[3]=in_range?1:0;o[4]=lo;o[5]=hi;
 o[6]=tp[0];o[7]=tp[8];o[8]=w;o[9]=first_valid?1:0;
}
void launch(int n,const float* c,const float* init,double* out,cudaStream_t stream){solve<<<1,3,0,stream>>>(n,c,init,out);}
'''
    cpp=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void launch(int,const float*,const float*,double*,cudaStream_t);
torch::Tensor forward(torch::Tensor c,torch::Tensor init){
 TORCH_CHECK(c.is_cuda()&&init.is_cuda()&&c.is_contiguous()&&init.is_contiguous(),"CUDA contiguous required");
 TORCH_CHECK(c.scalar_type()==torch::kFloat32&&init.scalar_type()==torch::kFloat32&&c.dim()==2&&c.size(1)==3&&init.numel()==4,"FP32 coefficients n,3 and init4 required");
 c10::cuda::CUDAGuard guard(c.device());auto out=torch::empty({3,65},c.options().dtype(torch::kFloat64));
 launch(c.size(0),c.data_ptr<float>(),init.data_ptr<float>(),out.data_ptr<double>(),at::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
'''
    config=Path(__file__).resolve().parents[1]/'submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h'
    return {'solver.cu':cu,'binding.cpp':cpp,'config.h':config.read_text(encoding='utf-8')}

def load(out):
    data=payloads();build=out/'generated';build.mkdir();h=hashlib.sha256(torch.__version__.encode())
    for name,text in sorted(data.items()):h.update(name.encode());h.update(text.encode());(build/name).write_text(text,encoding='utf-8')
    from torch.utils.cpp_extension import load
    mod=load(name='exact_outlier_solver_'+h.hexdigest()[:12],sources=[str(build/n) for n in ('binding.cpp','solver.cu')],extra_cflags=['-O3'],extra_cuda_cflags=FLAGS,build_directory=str(build),verbose=True)
    return mod
