#!/usr/bin/env python3
"""P1: multi-view depth inconsistency σ_z on frozen Gaussians (RESEARCH_PROTOCOL §3.2).

σ_z(s) = 1.4826 * median_i( MAD(z_i - z_{j→i}) ) on d_mask.

Usage:
  python scripts/measure_sigma_z.py \\
    --models ./output_compare/ggs_ablate_no_dn/scan24 \\
             ./output_compare/ggs_ablate_no_dn/scan105 \\
             ./output_compare/ggs_ablate_no_dn/scan122 \\
    --out ./output_compare/metrics/sigma_z_no_dn.json
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from random import Random

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render, sample_depth
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def _depth_residuals(
    gaussians: GaussianModel,
    render_pkg: dict,
    viewpoint_cam,
    nearest_cam,
    pipe,
    kernel_size: float,
    pixel_noise_th: float,
) -> torch.Tensor:
    """Return 1D tensor of (z_i - z_{j→i}) on geometric d_mask (same chain as PatchMatch)."""
    H = int(viewpoint_cam.image_height)
    W = int(viewpoint_cam.image_width)
    ix = (torch.arange(W, device="cuda", dtype=torch.float32) - viewpoint_cam.Cx) / viewpoint_cam.Fx
    iy = (torch.arange(H, device="cuda", dtype=torch.float32) - viewpoint_cam.Cy) / viewpoint_cam.Fy
    view_to_nearest_T = (
        -viewpoint_cam.world_view_transform[:3, :3].T @ nearest_cam.R @ nearest_cam.T
        + viewpoint_cam.world_view_transform[3, :3]
    )
    nearest_to_view_R = nearest_cam.R.transpose(1, 0) @ viewpoint_cam.world_view_transform[:3, :3]

    z_i = render_pkg["median_depth"].squeeze()
    depth_reshape = z_i.unsqueeze(-1)
    pts = torch.cat(
        [depth_reshape * ix[None, :, None], depth_reshape * iy[:, None, None], depth_reshape],
        dim=-1,
    )
    R = viewpoint_cam.R
    T = viewpoint_cam.T
    pts = (pts - T) @ R.T
    sampled_pkg = sample_depth(pts, nearest_cam, gaussians, pipe, kernel_size)
    pts_in_nearest_cam = sampled_pkg["sampled_depth"]
    pts_in_view_cam = view_to_nearest_T + pts_in_nearest_cam @ nearest_to_view_R
    z_j_to_i = pts_in_view_cam[..., 2]

    ix_i, iy_i = torch.meshgrid(
        torch.arange(W, device="cuda", dtype=torch.int32),
        torch.arange(H, device="cuda", dtype=torch.int32),
        indexing="xy",
    )
    pixels = torch.stack([ix_i, iy_i], dim=-1).type(torch.float32)
    pts_projections = pts_in_view_cam[..., :2] / torch.clamp_min(pts_in_view_cam[..., 2:], 1e-7)
    pts_projections = torch.addcmul(
        pts_projections.new_tensor([viewpoint_cam.Cx, viewpoint_cam.Cy]),
        pts_projections.new_tensor([viewpoint_cam.Fx, viewpoint_cam.Fy]),
        pts_projections,
    )
    pixel_noise = torch.pairwise_distance(pts_projections, pixels)
    d_mask = (
        sampled_pkg["inside"]
        & (pts_in_nearest_cam[..., -1] > 0.2)
        & (pts_in_view_cam[..., -1] > 0.2)
        & (pixel_noise < pixel_noise_th)
        & (z_i > 0)
    )
    if not d_mask.any():
        return torch.empty(0, device="cuda", dtype=torch.float32)
    return (z_i - z_j_to_i)[d_mask]


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def measure_one(model_path: Path, seed: int, pixel_noise_th: float) -> dict:
    # get_combined_args reads cfg_args from -m; isolate argv per model.
    old_argv = sys.argv
    sys.argv = ["measure_sigma_z.py", "-m", str(model_path)]
    try:
        parser = ArgumentParser(description="sigma_z")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        args = get_combined_args(parser)
        dataset = model.extract(args)
        pipe = pipeline.extract(args)
    finally:
        sys.argv = old_argv

    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    kernel_size = dataset.kernel_size
    bg = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    cams = scene.getTrainCameras()
    rng = Random(seed)

    view_mads: list[float] = []
    view_stats: list[dict] = []
    n_res_total = 0
    for cam in cams:
        if len(cam.nearest_id) == 0:
            continue
        nid = rng.choice(cam.nearest_id)
        nearest = cams[nid]
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipe, bg, kernel_size, require_depth=True)
            r = _depth_residuals(
                gaussians, render_pkg, cam, nearest, pipe, kernel_size, pixel_noise_th
            )
        r_np = r.detach().float().cpu().numpy()
        n_res_total += int(r_np.size)
        mad = _mad(r_np)
        view_mads.append(mad)
        view_stats.append(
            {
                "ref": cam.image_name,
                "nearest": nearest.image_name,
                "n": int(r_np.size),
                "mad": mad,
                "median_abs": float(np.median(np.abs(r_np))) if r_np.size else None,
            }
        )

    view_mads_arr = np.array([m for m in view_mads if np.isfinite(m)], dtype=np.float64)
    sigma_z = float(1.4826 * np.median(view_mads_arr)) if view_mads_arr.size else float("nan")
    scan = Path(model_path).name
    return {
        "scan": scan,
        "model_path": str(model_path),
        "seed": seed,
        "n_views_used": int(view_mads_arr.size),
        "n_residuals_total": n_res_total,
        "sigma_z": sigma_z,
        "median_view_mad": float(np.median(view_mads_arr)) if view_mads_arr.size else None,
        "pixel_noise_th": pixel_noise_th,
        "views": view_stats,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("./output_compare/metrics/sigma_z_no_dn.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pixel_noise_th", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true")
    # Dummy so ModelParams can be constructed in measure_one via get_combined_args
    args, _ = parser.parse_known_args()
    safe_state(args.quiet)

    reports = []
    for mp in args.models:
        print(f"[P1] measuring σ_z: {mp}")
        with torch.no_grad():
            rep = measure_one(mp, args.seed, args.pixel_noise_th)
        print(
            f"  {rep['scan']}: sigma_z={rep['sigma_z']:.6f} "
            f"n_views={rep['n_views_used']} n_res={rep['n_residuals_total']}"
        )
        reports.append(rep)

    # Freeze protocol metadata
    out = {
        "protocol": "RESEARCH_PROTOCOL §3.2",
        "formula": "sigma_z = 1.4826 * median_i(MAD(z_i - z_j_to_i on d_mask))",
        "seed": args.seed,
        "pixel_noise_th": args.pixel_noise_th,
        "view_policy": "all train views with non-empty nearest_id; one random nearest per view",
        "reports": reports,
        "sigma_z_by_scan": {r["scan"]: r["sigma_z"] for r in reports},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
