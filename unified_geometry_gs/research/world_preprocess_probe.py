"""Thin CUDA calls into archived production preprocessing, without rasterization.

Fixed conic/plane/normal upstreams isolate the local world-parameter VJP. The
four branches differ only in a normalization factor and/or local FP64 arithmetic.
No production sources, query grouping, root solving or loss weights are edited.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from research.precise_preprocess_backend import function_span, promote

ROOT = Path(__file__).resolve().parents[1]
RASTER = ROOT / 'submodules/diff-gaussian-rasterization'
VARIANTS = ('legacy32', 'normal32', 'legacy64', 'normal64', 'all32', 'all64')


def normal_fix(text):
    old = 'rnorm3df(normal_vector.x, normal_vector.y, normal_vector.z)'
    new = 'rnorm3df(cam_normal_vector.x, cam_normal_vector.y, cam_normal_vector.z)'
    if text.count(old) != 1:
        raise ValueError('Normalization source anchor changed.')
    return text.replace(old, new)


def ray_precision_fix(text):
    old = 'float dL_du_sigma           = dL_dray_len2_sigma_x2 * u;'
    new = ('float dL_du_sigma           = dL_dray_len2_sigma_x2 * u '
           '+ dL_drsigma * uvh_m.x / (rsigmat * ray_len2);')
    oldv = 'float dL_dv_sigma           = dL_dray_len2_sigma_x2 * v;'
    newv = ('float dL_dv_sigma           = dL_dray_len2_sigma_x2 * v '
            '+ dL_drsigma * uvh_m.y / (rsigmat * ray_len2);')
    if text.count(old)!=1 or text.count(oldv)!=1:
        raise ValueError('Ray-precision source anchors changed.')
    return text.replace(old,new).replace(oldv,newv)


def opacity_coef_fix(text):
    old='const float dL_ddet1     = -dL_ddet0 * coef;'
    if text.count(old)!=1:
        raise ValueError('Opacity-coefficient source anchor changed.')
    return text.replace(old,'const float dL_ddet1     = -dL_ddet0 * coef * coef;')


def all_fixes(text):
    return opacity_coef_fix(ray_precision_fix(normal_fix(text)))


def source_payloads():
    forward = (RASTER / 'cuda_rasterizer/render_forward.cu').read_text()
    backward = (RASTER / 'cuda_rasterizer/render_backward.cu').read_text()
    license_header = forward[:forward.index('#include')]
    def extract(text, marker):
        start, _, end = function_span(text, marker)
        return text[start:end]
    fwd = extract(forward, '__device__ __forceinline__ bool computeCov2D(')
    back = '\n\n'.join(extract(backward, marker) for marker in
        ('__device__ __forceinline__ void computeCov3D(', '__global__ void computeCov2DCUDA('))
    cuda = license_header + '''
#include "auxiliary.h"
#include "config.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
namespace probe_forward {
''' + fwd + '''
__global__ void run(int n, const float3* m, const glm::vec3* s, const float4* q,
 const float* opacity, const float* view, float fx, float fy, float tx, float ty,
 float kernel, float* out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 float c[3],coef; float3 normal; float4 plane;
 computeCov2D(m[i],fx,fy,tx,ty,kernel,nullptr,view,c,&normal,&plane,coef,s+i,q+i,1.f);
 float det=c[0]*c[2]-c[1]*c[1];
 float* o=out+11*i;
 o[0]=c[2]/det; o[1]=-c[1]/det; o[2]=c[0]/det; o[3]=coef*opacity[i];
 o[4]=plane.x; o[5]=plane.y; o[6]=plane.z; o[7]=plane.w;
 o[8]=normal.x; o[9]=normal.y; o[10]=normal.z;
}
}
void probe_fwd(int n,const float* m,const float* s,const float* q,const float* o,
 const float* v,float fx,float fy,float tx,float ty,float k,float* out,cudaStream_t stream){
 probe_forward::run<<<(n+127)/128,128,0,stream>>>(n,(const float3*)m,
 (const glm::vec3*)s,(const float4*)q,o,v,fx,fy,tx,ty,k,out);
}
'''
    for variant in VARIANTS:
        code = normal_fix(back) if variant.startswith('normal') else back
        if variant.startswith('all'):
            code = all_fixes(back)
        if variant.endswith('64'):
            code = promote(code)
        cuda += '\nnamespace ' + variant + ' {\n' + code + '\n}\n'
    cuda += '''
void probe_bwd(int mode,int n,const float* m,const int* r,const float* s,const float* q,
 const float* o,const float* v,float fx,float fy,float tx,float ty,float k,
 const float* gc,const float* gp,const float* gn,float* gm,float* gs,float* gq,
 float* go,cudaStream_t stream){
'''
    for i, variant in enumerate(VARIANTS):
        cuda += f'''if(mode=={i}) {variant}::computeCov2DCUDA<<<(n+127)/128,128,0,stream>>>(
 n,(const float3*)m,r,nullptr,(const glm::vec3*)s,(const float4*)q,o,1.f,fx,fy,tx,ty,k,v,
 (const float4*)gc,(const float4*)gp,(const glm::vec3*)gn,(glm::vec3*)gm,nullptr,
 (glm::vec3*)gs,(glm::vec4*)gq,go);
'''
    cuda += '}\n'
    cpp = '''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
void probe_fwd(int,const float*,const float*,const float*,const float*,const float*,
 float,float,float,float,float,float*,cudaStream_t);
void probe_bwd(int,int,const float*,const int*,const float*,const float*,const float*,const float*,
 float,float,float,float,float,const float*,const float*,const float*,float*,float*,float*,float*,cudaStream_t);
void check(torch::Tensor t){TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==torch::kFloat32,"CUDA contiguous float32 required");}
torch::Tensor forward(torch::Tensor m,torch::Tensor s,torch::Tensor q,torch::Tensor o,
 torch::Tensor v,double fx,double fy,double tx,double ty,double k){
 for(auto t:{m,s,q,o,v})check(t);
 TORCH_CHECK(m.dim()==2&&m.size(1)==3&&s.sizes()==m.sizes()&&q.size(0)==m.size(0)&&q.size(1)==4&&o.numel()==m.size(0)&&v.numel()==16,"Invalid shapes");
 c10::cuda::CUDAGuard guard(m.device());
 auto out=torch::empty({m.size(0),11},m.options());
 probe_fwd(m.size(0),m.data_ptr<float>(),s.data_ptr<float>(),q.data_ptr<float>(),o.data_ptr<float>(),v.data_ptr<float>(),fx,fy,tx,ty,k,out.data_ptr<float>(),at::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> backward(int mode,torch::Tensor m,torch::Tensor s,torch::Tensor q,
 torch::Tensor o,torch::Tensor v,torch::Tensor gc,torch::Tensor gp,torch::Tensor gn,
 double fx,double fy,double tx,double ty,double k){
 for(auto t:{m,s,q,o,v,gc,gp,gn})check(t);TORCH_CHECK(mode>=0&&mode<6,"Unknown branch");
 c10::cuda::CUDAGuard guard(m.device());auto r=torch::ones({m.size(0)},m.options().dtype(torch::kInt32));
 auto gm=torch::zeros_like(m),gs=torch::zeros_like(s),gq=torch::zeros_like(q),go=torch::zeros_like(o);
 probe_bwd(mode,m.size(0),m.data_ptr<float>(),r.data_ptr<int>(),s.data_ptr<float>(),q.data_ptr<float>(),o.data_ptr<float>(),v.data_ptr<float>(),fx,fy,tx,ty,k,gc.data_ptr<float>(),gp.data_ptr<float>(),gn.data_ptr<float>(),gm.data_ptr<float>(),gs.data_ptr<float>(),gq.data_ptr<float>(),go.data_ptr<float>(),at::cuda::getCurrentCUDAStream());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {gm,gs,gq,go};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);}
'''
    result = {'probe.cu': cuda, 'binding.cpp': cpp}
    for p in (RASTER / 'cuda_rasterizer').glob('*.h'):
        result[p.name] = p.read_text()
    return result


@lru_cache(maxsize=1)
def load_probe():
    payloads = source_payloads()
    h = hashlib.sha256(torch.__version__.encode())
    for name, value in sorted(payloads.items()):
        h.update(name.encode()); h.update(value.encode())
    # GLM contributes inline device definitions and must be included in identity.
    for p in sorted(x for x in (RASTER / 'third_party/glm/glm').rglob('*') if x.is_file()):
        h.update(p.relative_to(RASTER).as_posix().encode()); h.update(p.read_bytes())
    signature = h.hexdigest()[:12]
    build = ROOT / 'research/build' / f'world_probe_{signature}'
    build.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        p = build / name
        if not p.exists() or p.read_text() != value:
            p.write_text(value)
    from torch.utils.cpp_extension import load
    return load(name=f'uggs_world_probe_{signature}', sources=[str(build / n) for n in ('binding.cpp','probe.cu')],
        extra_include_paths=[str(build),str(RASTER/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda',
            '--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)
