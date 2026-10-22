"""Isolated sampling work grouping; five-pass roots and formulas stay unchanged.

Copies non-vendor sources/headers into a signature-specific build directory and
changes only SAMPLE_BATCH_SIZE. Original production files are never rewritten.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from research.median_cuda_backend import activate,load_backend as original_backend
ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=3)
def load_backend(variant='batch1'):
    if variant=='control':return original_backend('control')
    if variant not in ('batch1','batch4'):raise ValueError('Unknown sampling batch.')
    size=int(variant[-1])
    source=ROOT/'submodules/diff-gaussian-rasterization'
    payloads={}
    for pattern in ('*.cu','*.cpp','*.h','cuda_rasterizer/*.cu','cuda_rasterizer/*.h'):
        for p in source.glob(pattern):payloads[p.relative_to(source).as_posix()]=p.read_text()
    key='cuda_rasterizer/config.h'
    old='constexpr int SAMPLE_BATCH_SIZE   = 2;'
    if payloads[key].count(old)!=1:raise ValueError('Sample batch config anchor changed.')
    payloads[key]=payloads[key].replace(old,f'constexpr int SAMPLE_BATCH_SIZE   = {size};')
    h=hashlib.sha256(torch.__version__.encode())
    for name,text in sorted(payloads.items()):h.update(name.encode());h.update(text.encode())
    signature=h.hexdigest()[:12]
    build=ROOT/'research/build'/f'sampling_{variant}_{signature}'
    tree=build/'src'
    for name,text in payloads.items():
        p=tree/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists() or p.read_text()!=text:p.write_text(text)
    from torch.utils.cpp_extension import load
    files=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/render_forward.cu',
        'cuda_rasterizer/render_backward.cu','cuda_rasterizer/sample_forward.cu',
        'cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp']
    return load(name=f'uggs_sampling_{variant}_{signature}',sources=[str(tree/p) for p in files],
        extra_include_paths=[str(tree),str(tree/'cuda_rasterizer'),str(source),
            str(source/'cuda_rasterizer'),str(source/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda',
            '--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('control','batch1','batch4'),required=True)
    print(load_backend(parser.parse_args().variant).__file__,flush=True)
