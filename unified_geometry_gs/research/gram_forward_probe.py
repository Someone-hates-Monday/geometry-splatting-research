"""Isolated forward determinant diagnostic; not a queued training backend."""
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import torch
from research.world_preprocess_probe import RASTER,ROOT
from research.precise_preprocess_backend import function_span

def payloads():
    original=(RASTER/'cuda_rasterizer/render_forward.cu').read_text()
    start,op,end=function_span(original,'__device__ __forceinline__ bool computeCov2D(')
    f=original[start:end]
    changes={
        'float& coef,':'float& coef, float& gram_det,',
        'glm::mat3 cov;':'glm::mat3 cov;\n    float det_gram = 0.f;',
        'cov             = glm::transpose(M) * M;':'''cov             = glm::transpose(M) * M;
        // Cauchy-Binet: determinant of the 2x2 Gram block.
        glm::vec3 area = glm::cross(M[0], M[1]);
        det_gram = glm::dot(area, area);''',
        '    cov2D[0]          = float(cov[0][0] + kernel_size);':'''    if (!scale) det_gram = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
    gram_det = det_gram + kernel_size * (cov[0][0] + cov[1][1]) + kernel_size * kernel_size;
    cov2D[0]          = float(cov[0][0] + kernel_size);''',
        'const float det_0 = fmaxf(1e-6f, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);':'const float det_0 = fmaxf(1e-6f, det_gram);',
        'const float det_1 = fmaxf(1e-6f, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);':'const float det_1 = fmaxf(1e-6f, gram_det);'
    }
    for a,b in changes.items():
        if f.count(a)!=1:raise ValueError('Frozen forward source anchor changed: '+a)
        f=f.replace(a,b)
    # The kernel never exercises the null-scale fallback. Its branch is kept
    # unchanged except determinant output and is not certified by this probe.
    wrapper='''
__global__ void run(int n,const float3* m,const glm::vec3* s,const float4* q,
 const float* opacity,const float* view,float fx,float fy,float tx,float ty,float kernel,float* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 float c[3],coef,det;float3 normal;float4 plane;
 computeCov2D(m[i],fx,fy,tx,ty,kernel,nullptr,view,c,&normal,&plane,coef,det,s+i,q+i,1.f);
 float* o=out+11*i;
 o[0]=c[2]/det;o[1]=-c[1]/det;o[2]=c[0]/det;o[3]=coef*opacity[i];
 o[4]=plane.x;o[5]=plane.y;o[6]=plane.z;o[7]=plane.w;
 o[8]=normal.x;o[9]=normal.y;o[10]=normal.z;
}
}
void gram_fwd(int n,const float* m,const float* s,const float* q,const float* o,
 const float* v,float fx,float fy,float tx,float ty,float k,float* out,cudaStream_t stream){
 gram32::run<<<(n+127)/128,128,0,stream>>>(n,(const float3*)m,(const glm::vec3*)s,(const float4*)q,o,v,fx,fy,tx,ty,k,out);
}
'''
    cuda=original[:original.index('#include')]+'''\n#include "auxiliary.h"
#include "config.h"
namespace gram32 {
'''+f+wrapper
    cpp='''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void gram_fwd(int,const float*,const float*,const float*,const float*,const float*,float,float,float,float,float,float*,cudaStream_t);
torch::Tensor forward(torch::Tensor m,torch::Tensor s,torch::Tensor q,torch::Tensor o,torch::Tensor v,double fx,double fy,double tx,double ty,double k){
 for(auto t:{m,s,q,o,v})TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==torch::kFloat32,"CUDA contiguous float32 required");
 TORCH_CHECK(m.dim()==2&&m.size(1)==3&&s.sizes()==m.sizes()&&q.size(0)==m.size(0)&&q.size(1)==4&&o.numel()==m.size(0)&&v.numel()==16,"Shapes changed");
 c10::cuda::CUDAGuard guard(m.device());auto out=torch::empty({m.size(0),11},m.options());
 gram_fwd(m.size(0),m.data_ptr<float>(),s.data_ptr<float>(),q.data_ptr<float>(),o.data_ptr<float>(),v.data_ptr<float>(),fx,fy,tx,ty,k,out.data_ptr<float>(),at::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
'''
    result={'probe.cu':cuda,'binding.cpp':cpp}
    for p in (RASTER/'cuda_rasterizer').glob('*.h'):result[p.name]=p.read_text()
    return result

@lru_cache(maxsize=1)
def load_probe():
    data=payloads();h=hashlib.sha256(torch.__version__.encode())
    for n,v in sorted(data.items()):h.update(n.encode());h.update(v.encode())
    for p in sorted(x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()):h.update(p.relative_to(RASTER).as_posix().encode());h.update(p.read_bytes())
    sig=h.hexdigest()[:12];build=ROOT/'research/build'/f'gram_forward_{sig}';build.mkdir(parents=True,exist_ok=True)
    for n,v in data.items():
        p=build/n
        if not p.exists() or p.read_text()!=v:p.write_text(v)
    from torch.utils.cpp_extension import load
    return load(name=f'uggs_gram_forward_{sig}',sources=[str(build/n) for n in ('binding.cpp','probe.cu')],
        extra_include_paths=[str(build),str(RASTER/'third_party/glm')],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda','--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)
