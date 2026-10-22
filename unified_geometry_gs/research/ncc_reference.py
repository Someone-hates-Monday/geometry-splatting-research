"""Independent, differentiable reference for the repository's 7x7 half-step NCC.

Research diagnostics only: production kernels and historical recipes are unchanged.
R maps reference-camera coordinates to neighbour-camera coordinates (column vectors).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def statistics(x: torch.Tensor, y: torch.Tensor):
    xc = x - x.mean(-1, keepdim=True)
    yc = y - y.mean(-1, keepdim=True)
    a = xc.square().sum(-1)
    b = yc.square().sum(-1)
    c = (xc * yc).sum(-1)
    return a, b, c, xc, yc


def score(x: torch.Tensor, y: torch.Tensor, *, legacy_gradient=False):
    a, b, c, xc, yc = statistics(x, y)
    denominator = a * b + EPS
    s = c.square() / denominator
    if not legacy_gradient:
        return s
    # Same forward value, but reproduce the CUDA coefficient of dB/dy.
    g = 2 * c[..., None] / denominator[..., None] * xc
    g = g - 2 * s[..., None] / (b[..., None] + EPS) * yc
    return s.detach() + ((y - y.detach()) * g.detach()).sum(-1)


def _sample(image: torch.Tensor, uv: torch.Tensor):
    h, w = image.shape
    normalized = torch.stack((2 * uv[..., 0] / (w - 1) - 1,
                              2 * uv[..., 1] / (h - 1) - 1), -1)
    return F.grid_sample(image[None, None], normalized[None], mode="bilinear",
                         padding_mode="border", align_corners=True)[0, 0]


def warped_patches(depths, normals, uvs, rotation, translation, image_r, image_n,
                   intrinsics_r, intrinsics_n):
    """Return reference/warped intensities [N,49] and kernel-equivalent validity.

    Use float64 for auditing; validity is a hard, detached predicate as in CUDA.
    Intrinsics are (fx,fy,cx,cy). Normals need not be unit length, as in the kernel.
    """
    fx, fy, cx, cy = intrinsics_r
    fnx, fny, cnx, cny = intrinsics_n
    centers = uvs.to(depths.dtype)
    rays = torch.stack(((centers[:, 0] - cx) / fx,
                        (centers[:, 1] - cy) / fy, torch.ones_like(depths)), -1)
    distance = -(rays * normals).sum(-1) * depths
    homography = rotation[None] - translation[None, :, None] * normals[:, None, :] / distance[:, None, None]
    offsets = torch.arange(-3, 4, dtype=depths.dtype, device=depths.device) / 2
    ov, ou = torch.meshgrid(offsets, offsets, indexing="ij")
    uv_r = centers[:, None] + torch.stack((ou.flatten(), ov.flatten()), -1)[None]
    ray = torch.stack(((uv_r[..., 0] - cx) / fx,
                       (uv_r[..., 1] - cy) / fy, torch.ones_like(uv_r[..., 0])), -1)
    projected = torch.einsum("nij,nkj->nki", homography, ray)
    uv_n = torch.stack((fnx * projected[..., 0] / projected[..., 2] + cnx,
                        fny * projected[..., 1] / projected[..., 2] + cny), -1)
    h, w = image_r.shape
    hn, wn = image_n.shape
    valid_r = ((centers[:, 0] - 1.5 > 0) & (centers[:, 0] + 1.5 < w - 1)
               & (centers[:, 1] - 1.5 > 0) & (centers[:, 1] + 1.5 < h - 1))
    valid_n = ((uv_n[..., 0] - 1.5 > 0) & (uv_n[..., 0] + 1.5 < wn - 1)
               & (uv_n[..., 1] - 1.5 > 0) & (uv_n[..., 1] + 1.5 < hn - 1)).all(-1)
    x, y = _sample(image_r, uv_r), _sample(image_n, uv_n)
    a, b, _, _, _ = statistics(x, y)
    valid = valid_r & valid_n & (a > 5e-6) & (b > 5e-6)
    return x, y, valid.detach()
