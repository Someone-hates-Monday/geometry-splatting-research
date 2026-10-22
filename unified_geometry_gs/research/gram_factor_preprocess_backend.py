"""Separate Gram-forward + audited factor-backward backend; smoke only."""
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from research.factor_preprocess_backend import payloads as factor_payloads,kernel_identity as factor_identity
from research.gram_forward_probe import payloads as gram_payloads
from research.precise_preprocess_backend import function_span
ROOT=Path(__file__).resolve().parents[1];RASTER=ROOT/'submodules/diff-gaussian-rasterization'
def payloads():
    data=factor_payloads();original=data['cuda_rasterizer/render_forward.cu'];probe=gram_payloads()['probe.cu']
    marker='__device__ __forceinline__ bool computeCov2D('
    s,_,e=function_span(original,marker);ps,_,pe=function_span(probe,marker)
    forward=original[:s]+probe[ps:pe]+original[e:]
    changes={'float ceof;':'float ceof;\n    float gram_det;',
        'ray_planes + idx, ceof, scales + idx':'ray_planes + idx, ceof, gram_det, scales + idx',
        'float det = (cov.x * cov.z - cov.y * cov.y);':'float det = gram_det;'}
    for a,b in changes.items():
        if forward.count(a)!=1:raise ValueError('Full forward call anchor changed: '+a)
        forward=forward.replace(a,b)
    data['cuda_rasterizer/render_forward.cu']=forward;return data
@lru_cache(maxsize=1)
def load_backend(variant='gram_factor64'):
    if variant!='gram_factor64':raise ValueError('Unknown Gram variant.')
    data=payloads();h=hashlib.sha256(torch.__version__.encode())
    for n,v in sorted(data.items()):h.update(n.encode());h.update(v.encode())
    for p in sorted(x for x in (RASTER/'third_party/glm/glm').rglob('*') if x.is_file()):h.update(p.relative_to(RASTER).as_posix().encode());h.update(p.read_bytes())
    sig=h.hexdigest()[:12];build=ROOT/'research/build'/f'gram_factor64_{sig}';tree=build/'src'
    for n,v in data.items():
        p=tree/n;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.read_text()!=v:p.write_text(v)
    from torch.utils.cpp_extension import load
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu','cuda_rasterizer/render_backward.cu',
        'cuda_rasterizer/sample_forward.cu','cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    return load(name=f'uggs_gram_factor64_{sig}',sources=[str(tree/n) for n in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(RASTER),str(RASTER/'cuda_rasterizer'),str(RASTER/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda','--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)
def kernel_identity(module,variant='gram_factor64'):
    if variant!='gram_factor64':raise ValueError('Unknown Gram certificate.')
    record=factor_identity(module,'factor64');marker='__device__ __forceinline__ bool computeCov2D('
    full=(Path(module.__file__).parent/'src/cuda_rasterizer/render_forward.cu').read_text();probe=gram_payloads()['probe.cu']
    s,_,e=function_span(full,marker);ps,_,pe=function_span(probe,marker)
    if full[s:e]!=probe[ps:pe]:raise ValueError('Full Gram function differs from probe.')
    record['gram_forward_function_sha256']=hashlib.sha256(full[s:e].encode()).hexdigest();record['gram_forward_identical_function_bytes']=True
    return record
