"""Isolated forward-solver experiments; installed rasterizer is never replaced.

The hybrid keeps the historical first subdivision and validity gate. A Newton
correction followed by two local endpoint checks can replace the remaining four
subdivisions. Failed checks fall back to those exact historical subdivisions.
Endpoint checks have an engineering float32 margin, not an interval certificate.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=3)
def load_backend(variant="hybrid"):
    from torch.utils.cpp_extension import load
    if variant not in ("control", "four", "hybrid"):
        raise ValueError("Unknown isolated solver variant.")
    source = ROOT / "submodules/diff-gaussian-rasterization"
    forward_path = source / "cuda_rasterizer/render_forward.cu"
    forward = forward_path.read_text()
    if variant == "hybrid":
        anchor = "auto iter = [&] __device__(auto first_const) {"
        if forward.count(anchor) != 1 or forward.count("bool done   = !in_range;") != 1:
            raise RuntimeError("Forward source changed: re-audit replacement anchors.")
        forward = forward.replace(anchor, "auto iter = [&] __device__(auto first_const, bool skip = false) {")
        forward = forward.replace("bool done   = !in_range;", "bool done   = !in_range || skip;")
        start = forward.index("        iter(std::true_type{});")
        end = forward.index("\n    }\n    // All threads", start)
        forward = forward[:start] + (ROOT / "research/median_hybrid_forward.inc").read_text() + forward[end:]
    elif variant == "four":
        anchor = "for (int i = 0; i < SPLIT_ITERATIONS - 1; i++)"
        if forward.count(anchor) != 1:
            raise RuntimeError("Subdivision source changed.")
        forward = forward.replace(anchor, "for (int i = 0; i < SPLIT_ITERATIONS - 2; i++)")
    originals = [source / p for p in (
        "cuda_rasterizer/rasterizer_impl.cu", "cuda_rasterizer/render_backward.cu",
        "cuda_rasterizer/sample_forward.cu", "cuda_rasterizer/sample_backward.cu",
        "rasterize_points.cu", "ext.cpp")]
    dependency_files = originals + sorted((source / "cuda_rasterizer").glob("*.h"))
    digest = hashlib.sha256((forward + torch.__version__).encode())
    for path in dependency_files:
        digest.update(path.read_bytes())
    signature = digest.hexdigest()[:12]
    build = ROOT / "research/build" / f"median_{variant}_{signature}"
    build.mkdir(parents=True, exist_ok=True)
    generated = build / "render_forward.cu"
    if not generated.exists() or generated.read_text() != forward:
        generated.write_text(forward)
    return load(name=f"uggs_median_{variant}_{signature}",
        sources=[str(generated), *map(str, originals)],
        extra_include_paths=[str(source), str(source / "cuda_rasterizer"), str(source / "third_party/glm")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--extended-lambda",
                          "--expt-relaxed-constexpr", "--allow-unsupported-compiler"],
        build_directory=str(build), verbose=False)


@contextmanager
def activate(backend):
    """Keep a complete forward/backward transaction inside this context."""
    import diff_gaussian_rasterization as rasterizer
    original = rasterizer._C
    rasterizer._C = backend
    try:
        yield
    finally:
        rasterizer._C = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("control", "four", "hybrid"), default="hybrid")
    args = parser.parse_args()
    print(load_backend(args.variant).__file__, flush=True)
