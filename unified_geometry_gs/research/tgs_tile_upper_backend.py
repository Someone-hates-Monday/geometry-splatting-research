"""Causal diagnostic: change only continuous-coordinate exclusive upper tiles."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from tgs_native_diagnostic_backend import ROOT,SOURCE

OUT=ROOT/'experiments/tgs_radius_contraction_v1/tile_coverage/upper_fix_build'


def load_upper_fix():
    original=ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/generated'
    forward=(original/'forward.cu').read_text()
    for axis,block in [('x','BLOCK_X'),('y','BLOCK_Y')]:
        old=f'uint b{axis}1 = min(grid.{axis}, max(0, (uint)((intersect_{axis} + {block} - 1) / {block})));'
        new=f'uint b{axis}1 = (uint)fminf((float)grid.{axis}, fmaxf(0.0f, floorf(intersect_{axis} / {block}) + 1.0f));'
        assert forward.count(old)==2
        forward=forward.replace(old,new)
    generated=OUT/'generated';generated.mkdir(parents=True,exist_ok=True)
    for name,contents in [('forward.cu',forward),('backward.cu',(original/'backward.cu').read_text())]:
        p=generated/name
        if p.exists():assert p.read_text()==contents
        else:p.write_text(contents)
    native=SOURCE/'native'
    sources=[native/'cuda_rasterizer/rasterizer_impl.cu',generated/'forward.cu',generated/'backward.cu',
             native/'cuda_rasterizer/utils.cu',native/'rasterize_points.cu',native/'ext.cpp']
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',
                      TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    start=time.time()
    mod=load(name='tgs_diag_6d61f4c_tile_upper_v1',sources=[str(p) for p in sources],
             extra_include_paths=[str(native/'third_party/glm'),str(native/'cuda_rasterizer')],
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(OUT),verbose=True)
    receipt=dict(module_file=mod.__file__,module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        changes='only four upper-bound expressions; no derivative, lower-bound or support-cutoff change',
        build_or_load_seconds=time.time()-start,diagnostic_only=True,training=False)
    path=OUT/'build_receipt.json'
    if not path.exists():path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return mod
