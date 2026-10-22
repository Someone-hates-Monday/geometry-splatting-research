"""Four-pass primal with Halley and quadratic denominator refinement.

This targets the underlying implicit root derivative, not the exact derivative
of the finite subdivision program. Original GGGS also uses an implicit backward.
The refinement is guarded against peak crossing and denominator clipping.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import torch
from research.median_cuda_backend import activate, load_backend as original_backend

ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def load_backend(variant="adjquad4"):
    if variant != "adjquad4":
        return original_backend(variant)
    from torch.utils.cpp_extension import load
    source=ROOT/"submodules/diff-gaussian-rasterization"
    forward=(source/"cuda_rasterizer/render_forward.cu").read_text()
    anchor="for (int i = 0; i < SPLIT_ITERATIONS - 1; i++)"
    if forward.count(anchor)!=1:
        raise ValueError("Subdivision source changed.")
    forward=forward.replace(anchor,"for (int i = 0; i < SPLIT_ITERATIONS - 2; i++)")
    backward=(source/"cuda_rasterizer/render_backward.cu").read_text()
    changes={
        "        float dT_dtm    = 0.f;":
            "        float solver_T=1.f, solver_dR=0.f, solver_ddR=0.f, solver_peak_distance=1e30f;\n        float dT_dtm    = 0.f;",
        "                dT_dtm += -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;":
            (ROOT/"research/median_adjquad_accumulate.inc").read_text(),
        "        dL_dmt_dT_dtm = dL_dpixel_mt / fmaxf(-dT_dtm, 1e-7f);":
            (ROOT/"research/median_adjquad_refine.inc").read_text()+
            "\n        dL_dmt_dT_dtm = dL_dpixel_mt / fmaxf(-dT_dtm, 1e-7f);",
    }
    for old,new in changes.items():
        if backward.count(old)!=1:
            raise ValueError(f"Adjoint source anchor changed: {old[:60]}")
        backward=backward.replace(old,new)
    originals=[source/p for p in ("cuda_rasterizer/rasterizer_impl.cu",
        "cuda_rasterizer/sample_forward.cu","cuda_rasterizer/sample_backward.cu","rasterize_points.cu","ext.cpp")]
    h=hashlib.sha256((forward+backward+torch.__version__).encode())
    for p in originals+sorted((source/"cuda_rasterizer").glob("*.h")):
        h.update(p.read_bytes())
    signature=h.hexdigest()[:12]
    build=ROOT/"research/build"/f"median_adjquad4_{signature}"
    build.mkdir(parents=True,exist_ok=True)
    generated=[]
    for name,text in (("render_forward.cu",forward),("render_backward.cu",backward)):
        p=build/name
        if not p.exists() or p.read_text()!=text:
            p.write_text(text)
        generated.append(str(p))
    return load(name=f"uggs_median_adjquad4_{signature}",sources=[*generated,*map(str,originals)],
        extra_include_paths=[str(source),str(source/"cuda_rasterizer"),str(source/"third_party/glm")],
        extra_cflags=["-O3"],extra_cuda_cflags=["-O3","--use_fast_math","--extended-lambda",
            "--expt-relaxed-constexpr","--allow-unsupported-compiler"],build_directory=str(build),verbose=False)


if __name__=="__main__":
    print(load_backend().__file__,flush=True)
