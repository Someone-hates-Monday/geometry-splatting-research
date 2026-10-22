"""New isolated probe: combine local fixes, FP64 and direct inverse-factor VJP.

Frozen original probe and precision helpers remain unchanged. The new branch
avoids assuming a float32-normalized quaternion is exactly orthogonal when
mapping inverse-precision gradients through a covariance round trip.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import torch
from research.world_preprocess_probe import source_payloads as original_payloads,all_fixes,RASTER,ROOT
from research.precise_preprocess_backend import promote,function_span
from research.stable_inverse_backend import generate


def promote_inverse(text):
    start,op,end=function_span(text,'__device__ __forceinline__ void computeInvCov3D(')
    signature=text[start:op];body=text[op:end];protected={}
    pattern=r'^.*(?:glm::vec3\* dL_dscale|float4\* dL_drot).*$'
    def protect(match):
        key=f'PROTECTED_FACTOR_POINTER_{len(protected)}';protected[key]=match.group();return key
    body=re.sub(pattern,protect,body,flags=re.MULTILINE)
    if len(protected)!=2:raise ValueError('Inverse-helper output pointer anchors changed.')
    body=body.replace('glm::mat3','glm::dmat3').replace('glm::vec3','glm::dvec3').replace('glm::vec4','glm::dvec4')
    body=re.sub(r'\bfloat\b','double',body);body=re.sub(r'(?<=\d)[fF]\b','',body)
    for token,raw in protected.items():body=body.replace(token,raw)
    for old,new in (('const glm::vec3 scale','const glm::dvec3 scale'),('const glm::mat3& R','const glm::dmat3& R'),
                    ('const float* dL_dcov3D','const double* dL_dcov3D'),('const glm::vec3 dL_dr','const glm::dvec3 dL_dr')):
        if signature.count(old)!=1:raise ValueError('Inverse-helper signature changed: '+old)
        signature=signature.replace(old,new)
    return text[:start]+signature+body+text[end:]


def corrected_factor_backward():
    text=(RASTER/'cuda_rasterizer/render_backward.cu').read_text()
    return promote_inverse(promote(generate(all_fixes(text))))


def source_payloads():
    result=original_payloads();back=corrected_factor_backward();functions=[]
    for name in ('__device__ __forceinline__ void computeCov3D(',
                 '__device__ __forceinline__ void computeInvCov3D(','__global__ void computeCov2DCUDA('):
        start,_,end=function_span(back,name);functions.append(back[start:end])
    result['probe.cu']+='\nnamespace factor64 {\n'+'\n\n'.join(functions)+'\n}\n'+'''
void probe_dispatch_bwd(int mode,int n,const float* m,const int* r,const float* s,const float* q,
 const float* o,const float* v,float fx,float fy,float tx,float ty,float k,
 const float* gc,const float* gp,const float* gn,float* gm,float* gs,float* gq,
 float* go,cudaStream_t stream){
 if(mode<6){probe_bwd(mode,n,m,r,s,q,o,v,fx,fy,tx,ty,k,gc,gp,gn,gm,gs,gq,go,stream);return;}
 factor64::computeCov2DCUDA<<<(n+127)/128,128,0,stream>>>(n,(const float3*)m,r,nullptr,
 (const glm::vec3*)s,(const float4*)q,o,1.f,fx,fy,tx,ty,k,v,(const float4*)gc,
 (const float4*)gp,(const glm::vec3*)gn,(glm::vec3*)gm,nullptr,(glm::vec3*)gs,
 (glm::vec4*)gq,go);
}
'''
    cpp=result['binding.cpp']
    if cpp.count('mode<6')!=1 or cpp.count('probe_bwd(')!=2:raise ValueError('Frozen binding anchors changed.')
    result['binding.cpp']=cpp.replace('mode<6','mode<7').replace('probe_bwd(','probe_dispatch_bwd(')
    return result


@lru_cache(maxsize=1)
def load_probe():
    payload=source_payloads();h=hashlib.sha256(torch.__version__.encode())
    for n,v in sorted(payload.items()):h.update(n.encode());h.update(v.encode())
    for p in sorted(x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()):
        h.update(p.relative_to(RASTER).as_posix().encode());h.update(p.read_bytes())
    sig=h.hexdigest()[:12];build=ROOT/'research/build'/f'factor_probe_{sig}';build.mkdir(parents=True,exist_ok=True)
    for n,v in payload.items():
        p=build/n
        if not p.exists() or p.read_text()!=v:p.write_text(v)
    from torch.utils.cpp_extension import load
    return load(name=f'uggs_factor_probe_{sig}',sources=[str(build/n) for n in ('binding.cpp','probe.cu')],
        extra_include_paths=[str(build),str(RASTER/'third_party/glm')],extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda','--expt-relaxed-constexpr','--allow-unsupported-compiler'],
        build_directory=str(build),verbose=False)
