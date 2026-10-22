"""Isolated local-cotangent contraction of the exact piecewise radius chain."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from tgs_native_diagnostic_backend import ROOT,SOURCE
from tgs_backward_fixed_backend import once

OUT=ROOT/'experiments/tgs_radius_contraction_v1'


def source_transform(text):
    text=once(text,'float loss_points_y[MAX_NB_POINTS] = {0.0f};',
        'float loss_points_y[MAX_NB_POINTS] = {0.0f};\n     float log_radius_gradient = 0.0f;')
    text=once(text,'dL_dnormal_x += (-p1_conv.x) * dL_doffset;',
        'const float local_cx = points_xy_image[idx].x - p1_conv.x;\n'
        '        const float local_cy = points_xy_image[idx].y - p1_conv.y;\n'
        '        // Incoming Gc/B were consistently flipped above when needed.\n'
        '        log_radius_gradient -= normal.x*dL_dnormal_x + normal.y*dL_dnormal_y\n'
        '             + (normal.x*local_cx + normal.y*local_cy)*dL_doffset;\n'
        '        dL_dnormal_x += local_cx * dL_doffset;')
    text=once(text,'dL_dnormal_y += (-p1_conv.y) * dL_doffset;',
                   'dL_dnormal_y += local_cy * dL_doffset;')
    text=once(text,'atomicAdd(&dL_dsigma_factor[global_id], -dL_dC * sigma_pre * Cx);',
                   '// Gamma is reconstructed from edge cotangents in preprocessCUDA.')
    text=once(text,'dL_dsigma_factor[idx]*dlogr_x','log_radius_gradient*dlogr_x')
    text=once(text,'dL_dsigma_factor[idx]*dlogr_y','log_radius_gradient*dlogr_y')
    text=once(text,'float dL_dnormal_x = dL_dphi_x * pixf.x;',
                   'float dL_dnormal_x = dL_dphi_x * (pixf.x - collected_xy[j].x);')
    text=once(text,'float dL_dnormal_y = dL_dphi_x * pixf.y;',
                   'float dL_dnormal_y = dL_dphi_x * (pixf.y - collected_xy[j].y);')
    assert 'dL_dsigma_factor[' not in text
    return text


def load_contraction():
    assert json.loads((OUT/'symbolic_verification.json').read_text())['passed']
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for e in manifest['files']:assert hashlib.sha256((SOURCE/e['path']).read_bytes()).hexdigest()==e['sha256']
    previous=ROOT/'experiments/tgs_branch_aux_v1/piecewise_build/generated'
    assert hashlib.sha256((previous/'backward.cu').read_bytes()).hexdigest()=='1fab2a104cfc05850abd960da5f2f9503b4165240b6fc88ab7414b79e416992a'
    generated=OUT/'generated';build=OUT/'build';generated.mkdir(exist_ok=True);build.mkdir(exist_ok=True)
    contents={'forward.cu':(previous/'forward.cu').read_text(),
              'backward.cu':source_transform((previous/'backward.cu').read_text())}
    for name,content in contents.items():
        path=generated/name
        if path.exists():assert path.read_text()==content
        else:path.write_text(content)
    native=SOURCE/'native'
    sources=[native/'cuda_rasterizer/rasterizer_impl.cu',generated/'forward.cu',generated/'backward.cu',
             native/'cuda_rasterizer/utils.cu',native/'rasterize_points.cu',native/'ext.cpp']
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',
                      TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    started=time.time()
    mod=load(name='tgs_diag_6d61f4c_contraction_v1',sources=[str(p) for p in sources],
             extra_include_paths=[str(native/'third_party/glm'),str(native/'cuda_rasterizer')],
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(build),verbose=True)
    receipt=dict(module_file=mod.__file__,module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        protocol_sha256=hashlib.sha256((ROOT/'TGS_RADIUS_CONTRACTION_PROTOCOL_V1.md').read_bytes()).hexdigest(),
        forward_unchanged=True,temporary_allocations_unchanged=True,extra_gamma_atomic_removed=True,
        build_or_load_seconds=time.time()-started,torch=torch.__version__,cuda=torch.version.cuda,training=False)
    path=OUT/'build_receipt.json'
    if not path.exists():path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return mod
