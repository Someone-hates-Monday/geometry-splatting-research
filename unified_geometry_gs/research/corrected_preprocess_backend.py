"""Opt-in exact local chain rules, with a matched FP64 legacy comparator.

All forward kernels, five median subdivisions, sampling batch=2, NCC shader,
float32 contributor accumulation and loss code remain unchanged.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from research.median_cuda_backend import activate
from research.precise_preprocess_backend import load_backend as precision_backend,promote,function_span
from research.world_preprocess_probe import all_fixes,normal_fix,ray_precision_fix,opacity_coef_fix
ROOT=Path(__file__).resolve().parents[1]


def payloads(variant):
    source=ROOT/'submodules/diff-gaussian-rasterization';result={}
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for p in source.glob(pattern):result[p.relative_to(source).as_posix()]=p.read_text()
    key='cuda_rasterizer/render_backward.cu'
    transforms={'corrected64':all_fixes,'normal64':normal_fix,'ray64':ray_precision_fix,'coef64':opacity_coef_fix}
    if variant not in transforms:raise ValueError('Unknown correction variant.')
    result[key]=promote(transforms[variant](result[key]))
    return result


@lru_cache(maxsize=5)
def load_backend(variant='corrected64'):
    if variant=='legacy64':return precision_backend('precise2')
    result=payloads(variant)
    source=ROOT/'submodules/diff-gaussian-rasterization'
    h=hashlib.sha256(torch.__version__.encode())
    for n,v in sorted(result.items()):h.update(n.encode());h.update(v.encode())
    for p in sorted(x for x in (source/'third_party/glm/glm').rglob('*') if x.is_file()):
        h.update(p.relative_to(source).as_posix().encode());h.update(p.read_bytes())
    sig=h.hexdigest()[:12];build=ROOT/'research/build'/f'corrected_{variant}_{sig}';tree=build/'src'
    for name,value in result.items():
        p=tree/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.read_text()!=value:p.write_text(value)
    from torch.utils.cpp_extension import load
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu',
        'cuda_rasterizer/render_backward.cu','cuda_rasterizer/sample_forward.cu',
        'cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    return load(name=f'uggs_corrected_{variant}_{sig}',sources=[str(tree/n) for n in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(source),str(source/'cuda_rasterizer'),str(source/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda',
        '--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)


def kernel_identity(module,variant):
    """Check built full-backend function bytes against the independent probe recipe."""
    from research.world_preprocess_probe import source_payloads
    target=Path(module.__file__).parent/'src/cuda_rasterizer/render_backward.cu'
    actual=target.read_text();probe=source_payloads()['probe.cu']
    mode='legacy64' if variant=='legacy64' else 'all64'
    if variant not in ('legacy64','corrected64'):raise ValueError('Identity certificate only for audited paired variants.')
    probe=probe[probe.index('namespace '+mode+' {'):]
    cert={}
    for marker in ('__device__ __forceinline__ void computeCov3D(','__global__ void computeCov2DCUDA('):
        s,_,e=function_span(actual,marker);ps,_,pe=function_span(probe,marker)
        a,b=actual[s:e],probe[ps:pe]
        if a!=b:raise ValueError('Full backend differs from audited probe '+marker)
        cert[marker]=hashlib.sha256(a.encode()).hexdigest()
    return {'full_generated_backward_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'probe_recipe_function_sha256':cert,'identical_function_bytes':True}
