"""Float64 CPU transcription of GGGS affine ray parameters and contributor rules.

This is an independent diagnostic reference, not the installed CUDA backend.
CUDA parity must be checked before interpreting its measurements as production
statistics. In particular, float32 projection and tile-boundary decisions differ.
"""
from __future__ import annotations

import numpy as np
from plyfile import PlyData


def read_model(path):
    data = PlyData.read(path)["vertex"].data
    xyz = np.column_stack([data[x] for x in ("x", "y", "z")]).astype(float)
    scales2 = np.exp(2 * np.column_stack([data[f"scale_{i}"] for i in range(3)]).astype(float))
    filt2 = np.asarray(data["filter_3D"], dtype=float)**2
    smooth2 = scales2 + filt2[:, None]
    opacity = 1 / (1 + np.exp(-np.asarray(data["opacity"], dtype=float)))
    opacity *= np.sqrt(np.prod(scales2, axis=1) / np.prod(smooth2, axis=1))
    q = np.column_stack([data[f"rot_{i}"] for i in range(4)]).astype(float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    r, x, y, z = q.T
    rotation = np.empty((len(q), 3, 3))
    rotation[:, 0] = np.column_stack((1 - 2 * (y*y + z*z), 2 * (x*y - r*z), 2 * (x*z + r*y)))
    rotation[:, 1] = np.column_stack((2 * (x*y + r*z), 1 - 2 * (x*x + z*z), 2 * (y*z - r*x)))
    rotation[:, 2] = np.column_stack((2 * (x*z - r*y), 2 * (y*z + r*x), 1 - 2 * (x*x + y*y)))
    return xyz, smooth2, opacity, rotation


def project_model(model, camera, resolution=2, kernel=0.):
    xyz, scales2, opacity, rotation = model
    width = round(camera["width"] / resolution)
    height = round(camera["height"] / resolution)
    fx = camera["fx"] * width / camera["width"]
    fy = camera["fy"] * height / camera["height"]
    c2w = np.asarray(camera["rotation"])
    xyz_cam = (xyz - camera["position"]) @ c2w
    keep = xyz_cam[:, 2] > .2
    xyz_cam, scales2, opacity, rotation = xyz_cam[keep], scales2[keep], opacity[keep], rotation[keep]
    tc = np.linalg.norm(xyz_cam, axis=1)
    z = xyz_cam[:, 2]
    # Columns of B are world Gaussian axes expressed in camera coordinates.
    b = np.einsum("ij,njk->nik", c2w.T, rotation)
    cov = np.einsum("nik,nk,njk->nij", b, scales2, b)
    precision = np.einsum("nik,nk,njk->nij", b, 1 / scales2, b)
    u = np.clip(xyz_cam[:, 0] / z, -1.3 * width / (2 * fx), 1.3 * width / (2 * fx))
    v = np.clip(xyz_cam[:, 1] / z, -1.3 * height / (2 * fy), 1.3 * height / (2 * fy))
    jx = np.column_stack((fx / z, np.zeros_like(z), -fx * u / z))
    jy = np.column_stack((np.zeros_like(z), fy / z, -fy * v / z))
    covxx = np.einsum("ni,nij,nj->n", jx, cov, jx)
    covxy = np.einsum("ni,nij,nj->n", jx, cov, jy)
    covyy = np.einsum("ni,nij,nj->n", jy, cov, jy)
    det0 = np.maximum(1e-6, covxx * covyy - covxy**2)
    covxx, covyy = covxx + kernel, covyy + kernel
    det = covxx * covyy - covxy**2
    opacity = opacity * np.sqrt(det0 / np.maximum(1e-6, det))
    conic = np.column_stack((covyy / det, -covxy / det, covxx / det))
    mid = .5 * (covxx + covyy)
    radius = np.ceil(3 * np.sqrt(mid + np.sqrt(np.maximum(.1, mid**2 - det))))
    uv = np.column_stack((fx * xyz_cam[:, 0] / z + (width - 1) / 2,
                          fy * xyz_cam[:, 1] / z + (height - 1) / 2))
    grid = np.array([(width + 15) // 16, (height + 15) // 16])
    tile_min = np.minimum(grid, np.maximum(0, np.trunc((uv - radius[:, None]) / 16))).astype(int)
    tile_max = np.minimum(grid, np.maximum(0, np.trunc((uv + radius[:, None] + 15) / 16))).astype(int)
    uvh = np.column_stack((u, v, np.ones_like(z)))
    qm = np.einsum("nij,nj->ni", precision, uvh)
    vb = np.sum(qm * uvh, axis=1)
    ray_len2 = 1 + u*u + v*v
    factor = z / np.sqrt(ray_len2)
    px = ((v*v + 1) * qm[:, 0] - u*v * qm[:, 1] - u * qm[:, 2]) / vb
    py = (-u*v * qm[:, 0] + (u*u + 1) * qm[:, 1] - v * qm[:, 2]) / vb
    plane = np.column_stack((px * factor / fx, py * factor / fy, tc))
    widths = np.sqrt(ray_len2 / vb)
    keep = (det > 0) & np.all(tile_max > tile_min, axis=1)
    order = np.flatnonzero(keep)[np.argsort(tc[keep], kind="stable")]
    return {"uv": uv[order], "conic": conic[order], "opacity": opacity[order],
            "plane": plane[order], "widths": widths[order],
            "tile_min": tile_min[order], "tile_max": tile_max[order],
            "image_width": width, "image_height": height, "fx": fx, "fy": fy}


def ray_parameters(projected, pixel):
    tile = np.asarray(pixel) // 16
    keep = np.all((projected["tile_min"] <= tile) & (tile < projected["tile_max"]), axis=1)
    ids = np.flatnonzero(keep)
    diff = projected["uv"][ids] - pixel
    c = projected["conic"][ids]
    power = -.5 * (c[:, 0] * diff[:, 0]**2 + c[:, 2] * diff[:, 1]**2) - c[:, 1] * diff[:, 0] * diff[:, 1]
    alpha = np.minimum(.99, projected["opacity"][ids] * np.exp(np.minimum(power, 0)))
    keep = (power <= 0) & (alpha >= 1 / 255)
    ids, diff, alpha = ids[keep], diff[keep], alpha[keep]
    # Original rasterizer stops before the primitive that would take T below 1e-4.
    prefix = np.cumprod(1 - alpha)
    stop = int(np.searchsorted(-prefix, -1e-4, side="right"))
    ids, diff, alpha = ids[:stop], diff[:stop], alpha[:stop]
    plane = projected["plane"][ids]
    peaks = plane[:, 2] + (plane[:, :2] * diff).sum(axis=1)
    widths = projected["widths"][ids]
    init = 0.
    if len(alpha):
        trans_before = np.r_[1., np.cumprod(1 - alpha)[:-1]]
        init = float(peaks[np.flatnonzero(trans_before > .5)[-1]])
    return alpha, peaks, widths, init
