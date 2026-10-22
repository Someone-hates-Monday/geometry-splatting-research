"""Derivative-only isolated variant with explicit piecewise/subgradient rules.

Forward stays byte-identical to the prior SH-corrected variant. Exact cap
derivatives are an explicit option here, not an assertion about author intent
or a recommendation to change a training optimizer without a comparison.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from tgs_native_diagnostic_backend import ROOT,SOURCE
from tgs_backward_fixed_backend import transformed,once

OUT=ROOT/'experiments/tgs_branch_aux_v1/piecewise_build'


def source_transform(text):
    text=transformed(text,radius=True)
    text=once(text,'const float dL_dC = con_o.w * dL_dalpha;',
        '// Exact local chain rule through min(0.99, raw_alpha).\n'
        '             if (con_o.w * Cx >= 0.99f) dL_dalpha = 0.0f;\n'
        '             const float dL_dC = con_o.w * dL_dalpha;')
    text=once(text,'float dL_dphi_x = dL_dC * (sigma_pre / phi_x) * Cx;',
        'float dL_dphi_x = dL_dC * (sigma_pre / phi_x) * Cx;\n'
        '             bool selected_edge = false;')
    text=once(text,'if (fabsf(distances[k] - max_val) < 1e-6f) {',
        'if (!selected_edge && distances[k] == max_val) {\n'
        '                    // One exact maximizing edge: a valid Clarke choice.\n'
        '                    selected_edge = true;')
    pattern=r'if \(length > 1e-8f\) \{\s*cross_prod\.x /= length;\s*cross_prod\.y /= length;\s*cross_prod\.z /= length;\s*\}'
    text,count=re.subn(pattern,
        'const float normal_denom = fmaxf(length, 1e-4f);\n'
        '     cross_prod.x /= normal_denom;\n'
        '     cross_prod.y /= normal_denom;\n'
        '     cross_prod.z /= normal_denom;',text)
    assert count==1
    text=once(text,'float length_inv = 1 / length;',
        'float length_inv = 1 / normal_denom;\n'
        '     // Below the norm floor, d(v/c)/dv=I/c; there is no projection term.\n'
        '     if (length <= 1e-4f) {\n'
        '         for (int j=0; j<9; ++j) normal_transpose[j] = 0.0f;\n'
        '     }')
    return text


def load_piecewise():
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for e in manifest['files']:
        assert hashlib.sha256((SOURCE/e['path']).read_bytes()).hexdigest()==e['sha256']
    OUT.mkdir(parents=True,exist_ok=True)
    native=SOURCE/'native'
    paths=['cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/forward.cu',
           'cuda_rasterizer/backward.cu','cuda_rasterizer/utils.cu','rasterize_points.cu','ext.cpp']
    sources=[native/p for p in paths]
    forward=once(sources[1].read_text(),'(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)',
                 'glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)')
    backward=source_transform(sources[2].read_text())
    generated=OUT/'generated';generated.mkdir(exist_ok=True)
    for j,content in [(1,forward),(2,backward)]:
        path=generated/sources[j].name
        if path.exists():assert path.read_text()==content
        else:path.write_text(content)
        sources[j]=path
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',
                      TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    start=time.time()
    mod=load(name='tgs_diag_6d61f4c_piecewise_v1',sources=[str(p) for p in sources],
             extra_include_paths=[str(native/'third_party/glm'),str(native/'cuda_rasterizer')],
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(OUT),verbose=True)
    receipt=dict(source_commit=manifest['native_commit'],module_file=mod.__file__,
        module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        changes=['previous SH constructor and geometry chain-rule corrections','exact saturated-alpha local derivative',
                 'first exact maximizing edge as a valid max subgradient','piecewise normal norm-floor derivative'],
        forward_byte_identical_to_previous_shfix=True,build_or_load_seconds=time.time()-start,
        torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),training=False)
    path=OUT/'build_receipt.json'
    if not path.exists():path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return mod
