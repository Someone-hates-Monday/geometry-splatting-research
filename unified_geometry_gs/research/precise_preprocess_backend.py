"""Isolated FP64 intermediates in world-parameter backward preprocessing.

Forward, sampling grouping, contributor accumulators, IFT and output tensor
dtype remain float32. This diagnostic tests precision of the local world VJP;
it does not make the entire renderer double precision or deterministic.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import re
import torch
from research.median_cuda_backend import activate,load_backend as original_backend
ROOT=Path(__file__).resolve().parents[1]


def function_span(text,start_text):
    start=text.index(start_text);op=text.index('{',start);depth=1;i=op+1
    while depth:
        if text[i]=='{':depth+=1
        if text[i]=='}':depth-=1
        i+=1
    return start,op,i


def promote(text):
    # Process the later caller before the earlier helper to preserve positions.
    for name in ('__global__ void computeCov2DCUDA(', '__device__ __forceinline__ void computeCov3D('):
        start,op,end=function_span(text,name)
        signature=text[start:op];body=text[op:end]
        protected={}
        for raw in ('const glm::vec3* scale = scales + idx;',
                    'const float* cov3D = cov3Ds + 6 * idx;',
                    'glm::vec3* dL_dscale = dL_dscales + idx;'):
            if raw in body:
                token=f'PROTECTED_PTR_{len(protected)}';protected[token]=raw;body=body.replace(raw,token)
        body=body.replace('glm::mat3','glm::dmat3').replace('glm::vec3','glm::dvec3').replace('glm::vec2','glm::dvec2').replace('glm::vec4','glm::dvec4')
        body=re.sub(r'\bfloat\b','double',body)
        body=re.sub(r'(?<=\d)[fF]\b','',body)
        for raw,replacement in (('fmaxf(','fmax('),('fminf(','fmin('),('sqrtf(','sqrt('),
                                ('rsqrtf(','rsqrt('),('norm3df(','norm3d('),('rnorm3df(','rnorm3d(')):
            body=body.replace(raw,replacement)
        body=re.sub(r'__frcp_rn\((scale_local\[[012]\])\)',r'(1.0 / \1)',body)
        for token,raw in protected.items():body=body.replace(token,raw)
        body=body.replace('float4{dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w}',
            'float4{static_cast<float>(dL_dq.x), static_cast<float>(dL_dq.y), static_cast<float>(dL_dq.z), static_cast<float>(dL_dq.w)}')
        body=body.replace('float3 dL_dt = {t.x * rtc * dL_dtc, t.y * rtc * dL_dtc, t.z * rtc * dL_dtc};',
            'glm::dvec3 dL_dt(t.x * rtc * dL_dtc, t.y * rtc * dL_dtc, t.z * rtc * dL_dtc);')
        body=body.replace('{dL_dtx + dL_dt.x, dL_dty + dL_dt.y, dL_dtz + dL_dt.z}',
            '{static_cast<float>(dL_dtx + dL_dt.x), static_cast<float>(dL_dty + dL_dt.y), static_cast<float>(dL_dtz + dL_dt.z)}')
        if 'computeCov2DCUDA' in name:
            old='float3 t     = transformPoint4x3(mean, view_matrix);'
            if body.count(old)!=1:raise ValueError('World position anchor changed.')
            body=body.replace(old,'float3 t_input = transformPoint4x3(mean, view_matrix);\n    glm::dvec3 t(t_input.x, t_input.y, t_input.z);')
        else:
            signature=signature.replace('const glm::mat3& R','const glm::dmat3& R').replace('const glm::vec3 dL_dr','const glm::dvec3 dL_dr').replace('const float* dL_dcov3D','const double* dL_dcov3D')
            # scale is passed from the promoted local scale vector.
            signature=signature.replace('const glm::vec3 scale','const glm::dvec3 scale')
        text=text[:start]+signature+body+text[end:]
    return text


@lru_cache(maxsize=3)
def load_backend(variant='precise2'):
    if variant=='control':return original_backend('control')
    if variant not in ('precise2','precise1'):raise ValueError('Unknown precision variant.')
    source=ROOT/'submodules/diff-gaussian-rasterization';payloads={}
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for p in source.glob(pattern):payloads[p.relative_to(source).as_posix()]=p.read_text()
    payloads['cuda_rasterizer/render_backward.cu']=promote(payloads['cuda_rasterizer/render_backward.cu'])
    if variant=='precise1':
        key='cuda_rasterizer/config.h';old='constexpr int SAMPLE_BATCH_SIZE   = 2;'
        if payloads[key].count(old)!=1:raise ValueError('Batch config anchor changed.')
        payloads[key]=payloads[key].replace(old,'constexpr int SAMPLE_BATCH_SIZE   = 1;')
    h=hashlib.sha256(torch.__version__.encode())
    for name,value in sorted(payloads.items()):h.update(name.encode());h.update(value.encode())
    signature=h.hexdigest()[:12];build=ROOT/'research/build'/f'precise_{variant}_{signature}'
    tree=build/'src'
    for name,value in payloads.items():
        p=tree/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.read_text()!=value:p.write_text(value)
    from torch.utils.cpp_extension import load
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu',
        'cuda_rasterizer/render_backward.cu','cuda_rasterizer/sample_forward.cu',
        'cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    return load(name=f'uggs_precise_{variant}_{signature}',sources=[str(tree/p) for p in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(source),str(source/'cuda_rasterizer'),str(source/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda',
            '--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=('precise2','precise1'),default='precise2')
    print(load_backend(p.parse_args().variant).__file__,flush=True)
