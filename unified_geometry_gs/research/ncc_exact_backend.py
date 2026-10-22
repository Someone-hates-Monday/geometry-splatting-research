"""Opt-in exact NCC derivative in an isolated CUDA extension.

The forward score and validity logic are copied without changes. The installed
historical warp_patch_ncc package is never rebuilt or replaced. Build artifacts
stay under research/build. CUDA toolkit, a supported C++ compiler and Ninja are
required, as for the original extension.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = "float grad_variance_n = -output_ncc / (variance_n + 1e-8f);"
NEW = "float grad_variance_n = -output_ncc * variance_r / (variance_r * variance_n + 1e-8f);"


@lru_cache(maxsize=1)
def load_backend():
    from torch.utils.cpp_extension import load
    source = ROOT / "submodules/warp-patch-ncc"
    original = (source / "cuda_warp_patch_ncc/warp_patch_ncc_impl.cu").read_text(encoding="utf-8")
    if original.count(OLD) != 1:
        raise RuntimeError("NCC source changed: re-audit the derivative before building this prototype.")
    fixed = original.replace(OLD, NEW)
    signature = hashlib.sha256((fixed + torch.__version__).encode()).hexdigest()[:12]
    build = ROOT / "research/build" / f"ncc_exact_{signature}"
    build.mkdir(parents=True, exist_ok=True)
    generated = build / "warp_patch_ncc_impl.cu"
    if not generated.exists() or generated.read_text(encoding="utf-8") != fixed:
        generated.write_text(fixed, encoding="utf-8")
    return load(
        name=f"uggs_ncc_exact_{signature}",
        sources=[str(generated), str(source / "warp_patch_ncc.cu"), str(source / "ext.cpp")],
        extra_include_paths=[str(source), str(source / "cuda_warp_patch_ncc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--allow-unsupported-compiler"],
        build_directory=str(build), verbose=False,
    )


class _ExactNCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depths, normals, uvs, rotation, translation, image_r, image_n, intrinsics, debug):
        output, gd, gn, valid = load_backend().warp_patch_ncc(
            depths, normals, uvs, rotation.contiguous(), translation.contiguous(),
            image_r, image_n, *intrinsics, debug,
        )
        ctx.save_for_backward(gd, gn)
        return output, valid

    @staticmethod
    def backward(ctx, grad_output, grad_valid):
        gd, gn = ctx.saved_tensors
        return grad_output * gd, grad_output[:, None] * gn, None, None, None, None, None, None, None


def warp_patch_ncc(depths, normals, uvs, rotation, translation, image_r, image_n,
                   fx_r, fy_r, cx_r, cy_r, fx_n, fy_n, cx_n, cy_n, debug=False):
    return _ExactNCC.apply(depths, normals, uvs, rotation, translation, image_r, image_n,
                           (fx_r, fy_r, cx_r, cy_r, fx_n, fy_n, cx_n, cy_n), debug)


if __name__ == "__main__":
    backend = load_backend()
    print(f"Built isolated exact-gradient backend: {backend.__file__}")
