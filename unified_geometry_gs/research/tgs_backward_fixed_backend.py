"""Isolated derivative-only TGS variants; archived sources are never edited.

scale_only isolates the pixel Jacobian. geometry_fixed additionally restores
the inradius chain rule, using an existing zero-initialized temporary buffer.
Neither variant changes any forward arithmetic or existing native module.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from tgs_native_diagnostic_backend import SOURCE, ROOT

OUT = ROOT / 'experiments/tgs_backward_v1'


def once(source, old, new):
    assert source.count(old) == 1, (source.count(old), old[:90])
    return source.replace(old, new)


def transformed(backward, radius):
    backward = once(backward,
        '(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)',
        'glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)')
    anchor = '\t float3 dL_ddepht = {0.0f, 0.0f, dL_dmean2D[idx].x};'
    extra = ''
    if radius:
        extra = r'''
     // alpha = opacity * (-phi / r)^sigma, away from branch boundaries.
     // renderCUDA accumulated dL/d(log r), a scalar shared by all vertices.
     // Since r = abs(cross(B-A,C-A)) / perimeter, this differentiates exactly
     // the same inradius as forward's last oriented edge/incenter distance.
     float2 q[3];
     float edge_len[3];
     float perimeter = 0.0f;
     for (int i = 0; i < 3; ++i) q[i] = p_image[cumsum_for_triangle + i];
     for (int i = 0; i < 3; ++i) {
         const float dx = q[(i+1)%3].x - q[i].x;
         const float dy = q[(i+1)%3].y - q[i].y;
         edge_len[i] = sqrtf(dx*dx + dy*dy);
         perimeter += edge_len[i];
     }
     const float twice_area = (q[1].x-q[0].x)*(q[2].y-q[0].y)
                            -(q[1].y-q[0].y)*(q[2].x-q[0].x);
     for (int i = 0; i < 3; ++i) {
         const int next = (i+1)%3, prev = (i+2)%3;
         const float dperim_x = (q[i].x-q[next].x)/edge_len[i]
                             + (q[i].x-q[prev].x)/edge_len[prev];
         const float dperim_y = (q[i].y-q[next].y)/edge_len[i]
                             + (q[i].y-q[prev].y)/edge_len[prev];
         const float dlogr_x = (q[next].y-q[prev].y)/twice_area - dperim_x/perimeter;
         const float dlogr_y = (q[prev].x-q[next].x)/twice_area - dperim_y/perimeter;
         loss_points_x[indices[cumsum_for_triangle+i]] += dL_dsigma_factor[idx]*dlogr_x;
         loss_points_y[indices[cumsum_for_triangle+i]] += dL_dsigma_factor[idx]*dlogr_y;
     }
'''
        backward = once(backward,
            '\t\t\t const float dL_dC = con_o.w * dL_dalpha;',
            '\t\t\t const float dL_dC = con_o.w * dL_dalpha;\n'
            '             // Existing buffer was unused in the pinned implementation.\n'
            '             atomicAdd(&dL_dsigma_factor[global_id], -dL_dC * sigma_pre * Cx);')
    extra += r'''
     // p_image is in pixel units; perspective projection yields NDC units.
     // These factors apply to geometry only, not depth/normal/SH derivatives.
     for (int i = 0; i < 3; ++i) {
         loss_points_x[i] *= 0.5f * W;
         loss_points_y[i] *= 0.5f * H;
     }
'''
    return once(backward, anchor, extra + '\n' + anchor)


def load_fixed(radius=True):
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((SOURCE / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    variant = 'geometry_fixed' if radius else 'scale_only'
    target = OUT / variant
    generated, build = target / 'generated', target / 'build'
    generated.mkdir(parents=True, exist_ok=True)
    build.mkdir(parents=True, exist_ok=True)
    native = SOURCE / 'native'
    paths = ['cuda_rasterizer/rasterizer_impl.cu', 'cuda_rasterizer/forward.cu',
             'cuda_rasterizer/backward.cu', 'cuda_rasterizer/utils.cu',
             'rasterize_points.cu', 'ext.cpp']
    sources = [native / p for p in paths]
    fwd = sources[1].read_text().replace(
        '(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)',
        'glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)')
    bwd = transformed(sources[2].read_text(), radius)
    for j, content in [(1, fwd), (2, bwd)]:
        path = generated / sources[j].name
        if path.exists():
            assert path.read_text() == content, 'Do not silently overwrite a built variant'
        else:
            path.write_text(content)
        sources[j] = path
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8', CC='/usr/bin/gcc-11',
                      CXX='/usr/bin/g++-11', TORCH_CUDA_ARCH_LIST='12.0', MAX_JOBS='2')
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    start = time.time()
    mod = load(name='tgs_diag_6d61f4c_' + variant + '_v1',
               sources=[str(p) for p in sources],
               extra_include_paths=[str(native / 'third_party/glm'), str(native / 'cuda_rasterizer')],
               extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--use_fast_math'],
               build_directory=str(build), verbose=True)
    receipt = dict(variant=variant, source_commit=manifest['native_commit'],
                   sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   module_file=mod.__file__, module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
                   build_or_load_seconds=time.time() - start, torch=torch.__version__,
                   cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                   forward_same_as_shfix=True, original_archive_unchanged=True,
                   changes=['previous SH center constructor correction', 'W/2 and H/2 pixel Jacobian']
                           + (['complete inradius derivative'] if radius else []))
    (target / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    return mod

