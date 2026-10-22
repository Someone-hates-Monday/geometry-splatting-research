"""Peak-containing final cells get pass five; reuse the high-order adjoint.

An isolated development heuristic, not a root-error certificate or novelty
claim. All threads still participate in tile barriers during conditional work.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
from pathlib import Path
import torch
from research.median_adjquad_backend import activate,load_backend as parent_backend
ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def load_backend(variant='peakadj'):
    if variant!='peakadj':return parent_backend(variant)
    from torch.utils.cpp_extension import load
    source=ROOT/'submodules/diff-gaussian-rasterization'
    forward=(source/'cuda_rasterizer/render_forward.cu').read_text()
    changes={
        '        auto iter = [&] __device__(auto first_const) {':
            '        bool peak_in_final_cell=false;\n        auto iter = [&] __device__(auto first_const, auto last_const, bool skip=false) {\n            constexpr bool LAST=decltype(last_const)::value;\n            unsigned int peak_bins=0;',
        '            bool done   = !in_range;':'            bool done   = !in_range || skip;',
        '                    const bool ball    = rsigma > 0;':
            '                    const bool ball    = rsigma > 0;\n                    if constexpr (LAST) {\n                        if (ball && interval>0.f && t_peak>=depth_min && t_peak<=depth_max) {\n                            int bin=min(static_cast<int>(SPLIT)-1,max(0,static_cast<int>((t_peak-depth_min)/interval)));\n                            peak_bins |= 1u << bin;\n                            if (bin>0 && t_peak==depth_min+interval*bin) peak_bins |= 1u << (bin-1);\n                        }\n                    }',
        '            depth_max  = depth_min + (start_id + 1) * interval;\n            depth_min  = depth_min + (start_id + 0) * interval;\n            T_p[0]     = T_p[start_id];\n            T_p[SPLIT] = T_p[start_id + 1];':
            '            if constexpr (LAST) peak_in_final_cell=((peak_bins >> start_id) & 1u)!=0;\n            if (!skip) {\n                depth_max  = depth_min + (start_id + 1) * interval;\n                depth_min  = depth_min + (start_id + 0) * interval;\n                T_p[0]     = T_p[start_id];\n                T_p[SPLIT] = T_p[start_id + 1];\n            }',
        '        iter(std::true_type{});\n        for (int i = 0; i < SPLIT_ITERATIONS - 1; i++)\n            iter(std::false_type{});':
            '        iter(std::true_type{},std::false_type{});\n        for (int i=0;i<SPLIT_ITERATIONS-3;i++)\n            iter(std::false_type{},std::false_type{});\n        iter(std::false_type{},std::true_type{});\n        bool restore_five=peak_in_final_cell && in_range && inside;\n        int tile_restore_five=__syncthreads_or(restore_five);\n        if (tile_restore_five)\n            iter(std::false_type{},std::false_type{},!restore_five);',
    }
    for old,new in changes.items():
        if forward.count(old)!=1:raise ValueError(f'Peak adaptive source anchor changed: {old[:70]}')
        forward=forward.replace(old,new)
    backward=(source/'cuda_rasterizer/render_backward.cu').read_text()
    changes={
        '        float dT_dtm    = 0.f;':
            '        float solver_T=1.f, solver_dR=0.f, solver_ddR=0.f, solver_peak_distance=1e30f;\n        float dT_dtm    = 0.f;',
        '                dT_dtm += -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;':
            (ROOT/'research/median_adjquad_accumulate.inc').read_text(),
        '        dL_dmt_dT_dtm = dL_dpixel_mt / fmaxf(-dT_dtm, 1e-7f);':
            (ROOT/'research/median_adjquad_refine.inc').read_text()+'\n        dL_dmt_dT_dtm = dL_dpixel_mt / fmaxf(-dT_dtm, 1e-7f);',
    }
    for old,new in changes.items():
        if backward.count(old)!=1:raise ValueError(f'Backward source anchor changed: {old[:70]}')
        backward=backward.replace(old,new)
    originals=[source/p for p in ('cuda_rasterizer/rasterizer_impl.cu','cuda_rasterizer/sample_forward.cu',
        'cuda_rasterizer/sample_backward.cu','rasterize_points.cu','ext.cpp')]
    h=hashlib.sha256((forward+backward+torch.__version__).encode())
    for p in originals+sorted((source/'cuda_rasterizer').glob('*.h')):h.update(p.read_bytes())
    signature=h.hexdigest()[:12]
    build=ROOT/'research/build'/f'median_peakadj_{signature}';build.mkdir(parents=True,exist_ok=True)
    generated=[]
    for name,text in (('render_forward.cu',forward),('render_backward.cu',backward)):
        p=build/name
        if not p.exists() or p.read_text()!=text:p.write_text(text)
        generated.append(str(p))
    return load(name=f'uggs_median_peakadj_{signature}',sources=[*generated,*map(str,originals)],
        extra_include_paths=[str(source),str(source/'cuda_rasterizer'),str(source/'third_party/glm')],
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','--use_fast_math','--extended-lambda',
            '--expt-relaxed-constexpr','--allow-unsupported-compiler'],build_directory=str(build),verbose=False)


if __name__=='__main__':print(load_backend().__file__,flush=True)
