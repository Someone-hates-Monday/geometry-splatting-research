"""Full opt-in backend with the stress-audited direct inverse-factor FP64 VJP."""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from research.median_cuda_backend import activate
from research.factor_preprocess_probe import corrected_factor_backward,source_payloads as probe_payloads
from research.precise_preprocess_backend import function_span
ROOT=Path(__file__).resolve().parents[1]
RASTER=ROOT/'submodules/diff-gaussian-rasterization'


def payloads():
    result={}
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for p in RASTER.glob(pattern):result[p.relative_to(RASTER).as_posix()]=p.read_text()
    result['cuda_rasterizer/render_backward.cu']=corrected_factor_backward()
    return result


@lru_cache(maxsize=1)
def load_backend(variant='factor64'):
    if variant!='factor64':raise ValueError('Only the separately audited factor64 branch is supported.')
    data=payloads();h=hashlib.sha256(torch.__version__.encode())
    for n,v in sorted(data.items()):h.update(n.encode());h.update(v.encode())
    for p in sorted(x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()):h.update(p.relative_to(RASTER).as_posix().encode());h.update(p.read_bytes())
    sig=h.hexdigest()[:12];build=ROOT/'research/build'/f'factor64_{sig}';tree=build/'src'
    for n,v in data.items():
        p=tree/n;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.read_text()!=v:p.write_text(v)
    from torch.utils.cpp_extension import load
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu',
        'cuda_rasterizer/render_backward.cu','cuda_rasterizer/sample_forward.cu',
        'cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    return load(name=f'uggs_factor64_{sig}',sources=[str(tree/n) for n in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(RASTER),str(RASTER/'cuda_rasterizer'),str(RASTER/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda','--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)


def kernel_identity(module,variant='factor64'):
    if variant!='factor64':raise ValueError('Unknown factor certificate.')
    target=Path(module.__file__).parent/'src/cuda_rasterizer/render_backward.cu';actual=target.read_text()
    probe=probe_payloads()['probe.cu'];probe=probe[probe.index('namespace factor64 {'):];cert={}
    for marker in ('__device__ __forceinline__ void computeCov3D(',
                   '__device__ __forceinline__ void computeInvCov3D(','__global__ void computeCov2DCUDA('):
        s,_,e=function_span(actual,marker);ps,_,pe=function_span(probe,marker)
        a,b=actual[s:e],probe[ps:pe]
        if a!=b:raise ValueError('Full factor backend differs from audited probe.')
        cert[marker]=hashlib.sha256(a.encode()).hexdigest()
    return {'full_generated_backward_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'probe_recipe_function_sha256':cert,'identical_function_bytes':True}
