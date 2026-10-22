"""Isolated backward exposure counters, excluded from all latency comparisons.

Pinned SH-corrected control math is retained. Its otherwise unused temporary
factor buffer becomes P x 4 counters and is returned as a seventh result.
Each per-triangle count is <= H*W < 2^24, so float32 unit increments are exact.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from tgs_native_diagnostic_backend import ROOT,SOURCE
from tgs_backward_fixed_backend import once

OUT=ROOT/'experiments/tgs_branch_exposure_v1'


def load_counter():
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for e in manifest['files']:assert hashlib.sha256((SOURCE/e['path']).read_bytes()).hexdigest()==e['sha256']
    native=SOURCE/'native';generated=OUT/'generated';build=OUT/'build'
    generated.mkdir(parents=True,exist_ok=True);build.mkdir(exist_ok=True)
    forward=once((native/'cuda_rasterizer/forward.cu').read_text(),
                 '(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)',
                 'glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)')
    backward=once((native/'cuda_rasterizer/backward.cu').read_text(),
                 '(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)',
                 'glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)')
    counter=r'''
             // Counts of pairs actually traversed by this backward kernel.
             // No dependence on incoming gradient magnitude is imposed.
             atomicAdd(&dL_dsigma_factor[4*global_id+0], 1.0f);
             if (con_o.w*Cx >= 0.99f)
                 atomicAdd(&dL_dsigma_factor[4*global_id+1], 1.0f);
             int near_max_count=0, exact_max_count=0;
             for (int k=0;k<3;++k) {
                 near_max_count += fabsf(distances[k]-max_val)<1e-6f;
                 exact_max_count += distances[k]==max_val;
             }
             if (near_max_count>1) atomicAdd(&dL_dsigma_factor[4*global_id+2], 1.0f);
             if (exact_max_count>1) atomicAdd(&dL_dsigma_factor[4*global_id+3], 1.0f);
'''
    backward=once(backward,'const int global_id = collected_id[j];','const int global_id = collected_id[j];\n'+counter)
    wrapper=(native/'rasterize_points.cu').read_text();header=(native/'rasterize_points.h').read_text()
    old='std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>'
    new='std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>'
    wrapper=once(wrapper,old,new);header=once(header,old,new)
    wrapper=once(wrapper,'torch::Tensor dL_dsigma_factor = torch::zeros({P, 1}, triangles_points.options());',
                 'torch::Tensor dL_dsigma_factor = torch::zeros({P, 4}, triangles_points.options());')
    wrapper=once(wrapper,'return std::make_tuple(dL_dtriangle, dL_dsigma, dL_dcolors, dL_dopacity, dL_dsh, dL_dmeans2D);',
                 'return std::make_tuple(dL_dtriangle, dL_dsigma, dL_dcolors, dL_dopacity, dL_dsh, dL_dmeans2D, dL_dsigma_factor);')
    contents={'forward.cu':forward,'backward.cu':backward,'rasterize_points.cu':wrapper,
              'rasterize_points.h':header,'ext.cpp':(native/'ext.cpp').read_text()}
    for name,text in contents.items():
        path=generated/name
        if path.exists():assert path.read_text()==text
        else:path.write_text(text)
    sources=[native/'cuda_rasterizer/rasterizer_impl.cu',generated/'forward.cu',generated/'backward.cu',
             native/'cuda_rasterizer/utils.cu',generated/'rasterize_points.cu',generated/'ext.cpp']
    os.environ.update(CUDA_HOME='/usr/local/cuda-12.8',CC='/usr/bin/gcc-11',CXX='/usr/bin/g++-11',
                      TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']
    import torch
    from torch.utils.cpp_extension import load
    started=time.time()
    mod=load(name='tgs_diag_6d61f4c_counters_v1',sources=[str(p) for p in sources],
             extra_include_paths=[str(native),str(native/'third_party/glm'),str(native/'cuda_rasterizer')],
             extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math'],build_directory=str(build),verbose=True)
    receipt=dict(module_file=mod.__file__,module_sha256=hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        generated_header_sha256=hashlib.sha256((generated/'rasterize_points.h').read_bytes()).hexdigest(),
        source_commit=manifest['native_commit'],build_or_load_seconds=time.time()-started,
        counter_columns=['contributing_pairs','raw_alpha_ge_099','multiple_edges_within_1e_6','multiple_exact_max_edges'],
        counter_scope='actual backward render-kernel traversal, irrespective of upstream magnitude',
        timing_eligible=False,training=False)
    path=OUT/'build_receipt.json'
    if not path.exists():path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    return mod
