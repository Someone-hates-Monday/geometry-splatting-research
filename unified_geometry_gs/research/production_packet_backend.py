"""Copied full production preprocessing kernel; diagnostic extension only."""
from __future__ import annotations
import hashlib
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
RASTER=ROOT/'submodules/diff-gaussian-rasterization'
FLAGS=['-O3','--use_fast_math','--extended-lambda','--expt-relaxed-constexpr','--allow-unsupported-compiler']

def extract(text,marker):
    start=text.index(marker);op=text.index('{',start);i=op+1;depth=1
    while depth:
        depth+=(text[i]=='{')-(text[i]=='}');i+=1
    return text[start:i]

def payloads():
    src=(RASTER/'cuda_rasterizer/render_forward.cu').read_text(encoding='utf-8')
    code=src[:src.index('#include')]+'''
#include "auxiliary.h"
#include "config.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
'''
    for marker in ('__device__ __forceinline__ glm::vec3 computeColorFromSHSG(',
                   '__device__ __forceinline__ bool computeCov2D(',
                   'template <int C>\n__global__ void preprocessCUDA('):
        code+=extract(src,marker)+'\n'
    code+='''
__global__ void diagnostic_geometry(int n,const float3* m,const float* view,
 float tx,float ty,float* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float3 t=transformPoint4x3(m[i],view);float u=t.x/t.z,v=t.y/t.z;
 float limx=1.3f*tx,limy=1.3f*ty;
 float cx=fminf(limx,fmaxf(-limx,u))*t.z;
 float cy=fminf(limy,fmaxf(-limy,v))*t.z;
 float* o=out+7*i;o[0]=u;o[1]=v;o[2]=1.f;o[3]=cx/t.z;
 o[4]=cy/t.z;o[5]=1.f;o[6]=t.z;
}
void launch(int n,const float* means,const float* scales,const float* q,const float* o,
 const float* v,const float* p,const float* campos,const float* colors,int W,int H,
 float tx,float ty,float fx,float fy,int* radii,bool* clamped,float* uv,float* depths,
 float* plane,float* normals,float* rgb,float* conic,uint32_t* touched,bool* conditions,
 float* geom,cudaStream_t stream){
 dim3 grid((W+BLOCK_X-1)/BLOCK_X,(H+BLOCK_Y-1)/BLOCK_Y,1);
 preprocessCUDA<3><<<(n+255)/256,256,0,stream>>>(n,0,1,0,0,means,colors,o,
 (const glm::vec3*)scales,(const float4*)q,nullptr,nullptr,nullptr,nullptr,nullptr,1.f,
 v,p,(const glm::vec3*)campos,W,H,tx,ty,fx,fy,0.f,radii,clamped,(float2*)uv,depths,
 (float4*)plane,(float3*)normals,rgb,(float4*)conic,grid,touched,false,conditions);
 diagnostic_geometry<<<(n+127)/128,128,0,stream>>>(n,(const float3*)means,v,tx,ty,geom);
}
'''
    cpp='''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void launch(int,const float*,const float*,const float*,const float*,const float*,const float*,
 const float*,const float*,int,int,float,float,float,float,int*,bool*,float*,float*,float*,
 float*,float*,float*,uint32_t*,bool*,float*,cudaStream_t);
std::vector<torch::Tensor> forward(torch::Tensor m,torch::Tensor s,torch::Tensor q,
 torch::Tensor o,torch::Tensor v,torch::Tensor p,torch::Tensor campos,int W,int H,
 double tx,double ty,double fx,double fy){
 for(auto t:{m,s,q,o,v,p,campos})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==torch::kFloat32,"CUDA FP32 contiguous required");
 c10::cuda::CUDAGuard guard(m.device());int64_t n=m.size(0);
 auto radii=torch::zeros({n},m.options().dtype(torch::kInt32));
 auto clamped=torch::zeros({n,3},m.options().dtype(torch::kBool));
 auto conditions=torch::zeros({n},m.options().dtype(torch::kBool));
 auto touched=torch::zeros({n},m.options().dtype(torch::kInt32));
 auto uv=torch::full({n,2},NAN,m.options()),depths=torch::full({n},NAN,m.options());
 auto plane=torch::full({n,4},NAN,m.options()),normals=torch::full({n,3},NAN,m.options());
 auto rgb=torch::zeros({n,3},m.options()),conic=torch::full({n,4},NAN,m.options());
 auto geom=torch::full({n,7},NAN,m.options());
 launch(n,m.data_ptr<float>(),s.data_ptr<float>(),q.data_ptr<float>(),o.data_ptr<float>(),
 v.data_ptr<float>(),p.data_ptr<float>(),campos.data_ptr<float>(),rgb.data_ptr<float>(),W,H,
 tx,ty,fx,fy,radii.data_ptr<int>(),clamped.data_ptr<bool>(),uv.data_ptr<float>(),depths.data_ptr<float>(),
 plane.data_ptr<float>(),normals.data_ptr<float>(),rgb.data_ptr<float>(),conic.data_ptr<float>(),
 (uint32_t*)touched.data_ptr<int>(),conditions.data_ptr<bool>(),geom.data_ptr<float>(),at::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {plane,conic,uv,radii,depths,touched,geom};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
'''
    result={'probe.cu':code,'binding.cpp':cpp}
    for p in (RASTER/'cuda_rasterizer').glob('*.h'):result[p.name]=p.read_text(encoding='utf-8')
    return result

def load(out):
    generated=out/'generated';generated.mkdir()
    data=payloads()
    for name,value in data.items():(generated/name).write_text(value,encoding='utf-8')
    h=hashlib.sha256(torch.__version__.encode())
    for name,value in sorted(data.items()):h.update(name.encode());h.update(value.encode())
    signature=h.hexdigest()[:12]
    from torch.utils.cpp_extension import load
    mod=load(name=f'production_packet_{signature}',sources=[str(generated/x) for x in ('binding.cpp','probe.cu')],
        extra_include_paths=[str(generated),str(RASTER/'third_party/glm')],extra_cflags=['-O3'],
        extra_cuda_cflags=FLAGS,build_directory=str(generated),verbose=True)
    return mod
