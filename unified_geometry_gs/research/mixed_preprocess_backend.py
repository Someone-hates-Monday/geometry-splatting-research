"""Use FP64 world-backward intermediates only for filtered scale axis ratio >10.

Two disjoint kernels preserve register allocation of the float32 path. The
threshold is an engineering hypothesis, not a certified floating-point bound.
Original forward, contributor accumulation and five root subdivisions stay.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from research.median_cuda_backend import activate,load_backend as original_backend
from research.precise_preprocess_backend import promote,function_span,load_backend as precise_backend
ROOT=Path(__file__).resolve().parents[1]


def generate(original):
    precise=promote(original)
    hs,_,he=function_span(precise,'__device__ __forceinline__ void computeCov3D(')
    helper=precise[hs:he].replace('computeCov3D(', 'computeCov3DPrecise(')
    ks,_,ke=function_span(precise,'__global__ void computeCov2DCUDA(')
    kernel=precise[ks:ke].replace('computeCov2DCUDA(', 'computeCov2DPreciseCUDA(').replace('computeCov3D(', 'computeCov3DPrecise(')
    anchor='''    if (idx >= P || !(radii[idx] > 0))
        return;'''
    predicate='''
    bool use_precise = false;
    if (scales) {
        const glm::vec3 ss = scales[idx];
        const float smin = fminf(ss.x, fminf(ss.y, ss.z));
        const float smax = fmaxf(ss.x, fmaxf(ss.y, ss.z));
        use_precise = smin > 0.f && smax > 10.f * smin;
    }
'''
    if kernel.count(anchor)!=1:raise ValueError('Precision predicate anchor changed.')
    kernel=kernel.replace(anchor,anchor+predicate+'    if (!use_precise) return;\n')
    fs,_,fe=function_span(original,'__global__ void computeCov2DCUDA(')
    float_kernel=original[fs:fe]
    if float_kernel.count(anchor)!=1:raise ValueError('Float predicate anchor changed.')
    float_kernel=float_kernel.replace(anchor,anchor+predicate+'    if (use_precise) return;\n')
    result=original[:fs]+helper+'\n'+float_kernel+'\n'+kernel+original[fe:]
    launch='computeCov2DCUDA<<<(P + 255) / 256, 256>>>('
    if result.count(launch)!=1:raise ValueError('Covariance launcher anchor changed.')
    start=result.index(launch);end=result.index(');',start)+2
    call=result[start:end]
    result=result[:end]+'\n    '+call.replace('computeCov2DCUDA<<<','computeCov2DPreciseCUDA<<<')+result[end:]
    return result


@lru_cache(maxsize=3)
def load_backend(variant='mixed1'):
    if variant=='control':return original_backend('control')
    if variant=='precise2':return precise_backend('precise2')
    if variant!='mixed1':raise ValueError('Unknown mixed variant.')
    source=ROOT/'submodules/diff-gaussian-rasterization';payloads={}
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for p in source.glob(pattern):payloads[p.relative_to(source).as_posix()]=p.read_text()
    payloads['cuda_rasterizer/render_backward.cu']=generate(payloads['cuda_rasterizer/render_backward.cu'])
    key='cuda_rasterizer/config.h';old='constexpr int SAMPLE_BATCH_SIZE   = 2;'
    if payloads[key].count(old)!=1:raise ValueError('Batch config anchor changed.')
    payloads[key]=payloads[key].replace(old,'constexpr int SAMPLE_BATCH_SIZE   = 1;')
    h=hashlib.sha256(torch.__version__.encode())
    for name,value in sorted(payloads.items()):h.update(name.encode());h.update(value.encode())
    signature=h.hexdigest()[:12];build=ROOT/'research/build'/f'mixed_{signature}';tree=build/'src'
    for name,value in payloads.items():
        p=tree/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.read_text()!=value:p.write_text(value)
    from torch.utils.cpp_extension import load
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu',
        'cuda_rasterizer/render_backward.cu','cuda_rasterizer/sample_forward.cu',
        'cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    return load(name=f'uggs_mixed_{signature}',sources=[str(tree/p) for p in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(source),str(source/'cuda_rasterizer'),str(source/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda',
            '--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)


if __name__=='__main__':
    print(load_backend().__file__,flush=True)
