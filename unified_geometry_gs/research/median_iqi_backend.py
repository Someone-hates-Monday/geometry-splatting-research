"""Four subdivision passes followed by inverse quadratic interpolation.

Reuses three existing T samples in the final pass, without another contributor
traversal. Classical interpolation, not a novelty claim or a numerical certificate.
The installed package and original production sources remain unchanged.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import torch
from research.median_cuda_backend import activate, load_backend as original_backend

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=4)
def load_backend(variant="iqi4"):
    if variant != "iqi4":
        return original_backend(variant)
    from torch.utils.cpp_extension import load
    source = ROOT / "submodules/diff-gaussian-rasterization"
    text = (source / "cuda_rasterizer/render_forward.cu").read_text()
    changes = {
        "auto iter = [&] __device__(auto first_const) {":
            "float inverse_quad_depth = 0.f;\n        auto iter = [&] __device__(auto first_const, auto last_const) {",
        "            depth_max  = depth_min + (start_id + 1) * interval;":
            (ROOT / "research/median_iqi_final.inc").read_text() +
            "\n            depth_max  = depth_min + (start_id + 1) * interval;",
        "        iter(std::true_type{});\n        for (int i = 0; i < SPLIT_ITERATIONS - 1; i++)\n            iter(std::false_type{});":
            "        iter(std::true_type{}, std::false_type{});\n        for (int i = 0; i < SPLIT_ITERATIONS - 3; i++)\n            iter(std::false_type{}, std::false_type{});\n        iter(std::false_type{}, std::true_type{});",
        "mDepth      = in_range ? w_max * depth_max + w_min * depth_min : 0.f;":
            "mDepth      = in_range ? inverse_quad_depth : 0.f;",
    }
    for old, new in changes.items():
        if text.count(old) != 1:
            raise ValueError(f"IQI source anchor changed: {old[:60]}")
        text = text.replace(old, new)
    originals = [source / p for p in ("cuda_rasterizer/rasterizer_impl.cu",
        "cuda_rasterizer/render_backward.cu", "cuda_rasterizer/sample_forward.cu",
        "cuda_rasterizer/sample_backward.cu", "rasterize_points.cu", "ext.cpp")]
    h = hashlib.sha256((text+torch.__version__).encode())
    for path in originals + sorted((source/"cuda_rasterizer").glob("*.h")):
        h.update(path.read_bytes())
    signature = h.hexdigest()[:12]
    build = ROOT / "research/build" / f"median_iqi4_{signature}"
    build.mkdir(parents=True, exist_ok=True)
    generated = build / "render_forward.cu"
    if not generated.exists() or generated.read_text() != text:
        generated.write_text(text)
    return load(name=f"uggs_median_iqi4_{signature}", sources=[str(generated),*map(str,originals)],
        extra_include_paths=[str(source),str(source/"cuda_rasterizer"),str(source/"third_party/glm")],
        extra_cflags=["-O3"], extra_cuda_cflags=["-O3","--use_fast_math","--extended-lambda",
            "--expt-relaxed-constexpr","--allow-unsupported-compiler"],
        build_directory=str(build),verbose=False)


if __name__ == "__main__":
    print(load_backend().__file__,flush=True)
