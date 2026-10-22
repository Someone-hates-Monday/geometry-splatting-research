"""Verify NCC calculus, then quantify its relevance on frozen DTU models.

Examples (run with the existing uggs environment):
  python research/audit_ncc.py --out research/results/ncc_synthetic.json
  python research/audit_ncc.py --model output_compare/ggs_pipeline/scan37 \
    --views 6 --samples 2048 --out research/results/ncc_scan37.json

No training, kernel rebuild, or modification of source model metadata is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.ncc_reference import EPS, score, statistics, warped_patches


def quantiles(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {k: float(v) for k, v in zip(("min", "p50", "p90", "p99", "max"),
                                        np.quantile(x, [0, .5, .9, .99, 1]))}


def synthetic_checks():
    import sympy as sp
    a, b, c, e = sp.symbols("A B C epsilon", positive=True)
    s = c**2 / (a * b + e)
    exact = sp.diff(s, b)
    assert sp.simplify(exact + s * a / (a * b + e)) == 0
    relative_coefficient_error = sp.factor((-s / (b + e)) / exact - 1)
    generator = torch.Generator().manual_seed(731)
    x = torch.randn(12, 49, generator=generator, dtype=torch.float64) * .001
    y = (x * .8 + torch.randn(12, 49, generator=generator, dtype=torch.float64) * .0003).requires_grad_()
    exact_g, = torch.autograd.grad(score(x, y).sum(), y)
    legacy_g, = torch.autograd.grad(score(x, y, legacy_gradient=True).sum(), y)
    direction = torch.randn(y.shape, generator=generator, dtype=y.dtype)
    step = 1e-8
    finite_difference = (score(x, y.detach() + step * direction) -
                         score(x, y.detach() - step * direction)) / (2 * step)
    exact_direction = (exact_g * direction).sum(-1)
    legacy_direction = (legacy_g * direction).sum(-1)
    exact_error = (finite_difference - exact_direction).abs().max().item()
    legacy_error = (finite_difference - legacy_direction).abs().max().item()
    assert exact_error < 1e-4, exact_error
    assert legacy_error > 1, legacy_error
    # Positive and negative affine gains have exactly the same stabilized score.
    assert torch.allclose(score(x, x), score(x, -x), atol=1e-14, rtol=1e-14)
    # A threshold greater than one includes every finite nonnegative NCC score.
    assert ((1 - score(x, y.detach())).clamp(0, 2) < 1.2).all()
    return {
        "symbolic_dS_dB": str(exact),
        "legacy_relative_coefficient_error": str(relative_coefficient_error),
        "directional_derivative_max_abs_error_exact": exact_error,
        "directional_derivative_max_abs_error_legacy": legacy_error,
        "negative_gain_indistinguishable": True,
        "tau_1_2_is_no_correlation_filter_for_finite_nonnegative_score": True,
        "scope": "Algebra and deliberately low-contrast synthetic patches; not reconstruction quality.",
    }


def candidates(gaussians, render_pkg, cam, near, pipe, kernel_size):
    from gaussian_renderer import sample_depth
    h, w = cam.image_height, cam.image_width
    iy, ix = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")
    pixels = torch.stack((ix, iy), -1)
    z = render_pkg["median_depth"].squeeze()
    points = torch.stack((z * (ix - cam.Cx) / cam.Fx, z * (iy - cam.Cy) / cam.Fy, z), -1)
    world = (points - cam.T) @ cam.R.T
    sampled = sample_depth(world, near, gaussians, pipe, kernel_size)
    nr = near.R.T @ cam.world_view_transform[:3, :3]
    nt = -cam.world_view_transform[:3, :3].T @ near.R @ near.T + cam.world_view_transform[3, :3]
    back = nt + sampled["sampled_depth"] @ nr
    uv = back[..., :2] / back[..., 2:].clamp_min(1e-7)
    uv = uv * uv.new_tensor([cam.Fx, cam.Fy]) + uv.new_tensor([cam.Cx, cam.Cy])
    noise = torch.pairwise_distance(uv, pixels.float())
    mask = sampled["inside"] & (sampled["sampled_depth"][..., -1] > .2) & (back[..., -1] > .2) & (noise < 1) & (z > 0)
    return pixels[mask], z[mask], render_pkg["normal"].permute(1, 2, 0)[mask]


def audit_model(args):
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    import warp_patch_ncc
    corrected_backend = None
    if args.exact_backend:
        from research import ncc_exact_backend as corrected_backend
        corrected_backend.load_backend()

    model_path = args.model.resolve()
    old_argv = sys.argv
    try:
        sys.argv = ["audit_ncc.py", "-m", str(model_path)]
        parser = argparse.ArgumentParser()
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        combined = get_combined_args(parser)
        dataset, pipe = model.extract(combined), pipeline.extract(combined)
    finally:
        sys.argv = old_argv

    rng = random.Random(args.seed)
    all_rows = []
    arrays = {k: [] for k in ("coefficient_relative_error", "score_abs_error", "depth_gradient_relative_error", "normal_gradient_relative_error")}
    if corrected_backend:
        arrays.update({"cuda_legacy_depth_abs_error": [], "cuda_corrected_depth_abs_error": []})
    accepted_total = negative_total = valid_total = sampled_total = 0
    a_total = eps_large_total = cuda_valid_mismatch = 0
    cuda_accepted_total = cuda_negative_total = cuda_score_over_one = 0
    finite_difference_report = None
    # Scene regenerates multi_view.json even when loading a model. Redirect that
    # metadata to a temporary model directory, with a read-only use of original PLYs.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ncc-scene-", dir=args.out.parent.resolve()) as temporary:
        scratch = Path(temporary)
        (scratch / "point_cloud").symlink_to(model_path / "point_cloud", target_is_directory=True)
        dataset.model_path = str(scratch)
        gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        cameras = scene.getTrainCameras()
        bg = torch.tensor([1., 1., 1.], device="cuda")
        chosen = np.linspace(0, len(cameras) - 1, min(args.views, len(cameras)), dtype=int)
        for view_id in chosen:
            cam = cameras[int(view_id)]
            if not cam.nearest_id:
                continue
            near = cameras[rng.choice(cam.nearest_id)]
            with torch.no_grad():
                rendered = render(cam, gaussians, pipe, bg, dataset.kernel_size, require_depth=True)
                uv, depths, normals = candidates(gaussians, rendered, cam, near, pipe, dataset.kernel_size)
                if not len(uv):
                    continue
                indices = rng.sample(range(len(uv)), min(args.samples, len(uv)))
                uv, depths, normals = uv[indices].contiguous().int(), depths[indices], normals[indices]
                normals = torch.nn.functional.normalize(normals, dim=-1)
                rotation = near.world_view_transform[:3, :3].T @ cam.world_view_transform[:3, :3]
                translation = -rotation @ cam.world_view_transform[3, :3] + near.world_view_transform[3, :3]
                intr_r = (cam.Fx, cam.Fy, cam.Cx, cam.Cy)
                intr_n = (near.Fx, near.Fy, near.Cx, near.Cy)
            d = depths.double().detach().requires_grad_()
            n = normals.double().detach().requires_grad_()
            x, y, valid = warped_patches(d, n, uv, rotation.double(), translation.double(),
                                         cam.gray_image.squeeze().double(), near.gray_image.squeeze().double(), intr_r, intr_n)
            s = score(x, y)
            exact_d, exact_n = torch.autograd.grad(s.sum(), (d, n), retain_graph=True)
            legacy_d, legacy_n = torch.autograd.grad(score(x, y, legacy_gradient=True).sum(), (d, n))
            with torch.no_grad():
                a, b, c, _, _ = statistics(x, y)
                accepted = valid & ((1 - s).clamp(0, 2) < .9)
                neg = accepted & (c < 0)
                coefficient_error = ((a * b + EPS) / (a * (b + EPS)) - 1).abs()
                cuda_d = depths.detach().requires_grad_()
                cuda_n = normals.detach().requires_grad_()
            with torch.enable_grad():
                cuda_s, cuda_valid = warp_patch_ncc.warp_patch_ncc(cuda_d, cuda_n, uv, rotation.T.contiguous(), translation,
                                                                cam.gray_image.squeeze(), near.gray_image.squeeze(), *intr_r, *intr_n, False)
                cuda_gd, cuda_gn = torch.autograd.grad(cuda_s.sum(), (cuda_d, cuda_n))
                if corrected_backend:
                    fixed_d, fixed_n = depths.detach().requires_grad_(), normals.detach().requires_grad_()
                    fixed_s, fixed_valid = corrected_backend.warp_patch_ncc(
                        fixed_d, fixed_n, uv, rotation.T.contiguous(), translation,
                        cam.gray_image.squeeze(), near.gray_image.squeeze(), *intr_r, *intr_n, False)
                    fixed_gd, = torch.autograd.grad(fixed_s.sum(), (fixed_d,))
                    assert torch.equal(fixed_s, cuda_s) and torch.equal(fixed_valid, cuda_valid)
            common = valid & cuda_valid
            cuda_accepted = cuda_valid & ((1 - cuda_s).clamp(0, 2) < .9)
            cuda_accepted_total += int(cuda_accepted.sum())
            cuda_negative_total += int((cuda_accepted & (c < 0)).sum())
            cuda_score_over_one += int(((cuda_s > 1) & cuda_valid).sum())
            if corrected_backend:
                oracle_accepted = common & cuda_accepted
                arrays["cuda_legacy_depth_abs_error"].extend((cuda_gd.double() - exact_d).abs()[oracle_accepted].detach().cpu().tolist())
                arrays["cuda_corrected_depth_abs_error"].extend((fixed_gd.double() - exact_d).abs()[oracle_accepted].detach().cpu().tolist())
            for key, tensor in (
                ("coefficient_relative_error", coefficient_error[valid]),
                ("score_abs_error", (cuda_s.double() - s).abs()[common]),
                ("depth_gradient_relative_error", ((legacy_d - exact_d).abs() / exact_d.abs().clamp_min(1e-8))[accepted]),
                ("normal_gradient_relative_error", ((legacy_n - exact_n).norm(dim=-1) / exact_n.norm(dim=-1).clamp_min(1e-8))[accepted]),
            ):
                arrays[key].extend(tensor.detach().cpu().tolist())
            sampled_total += len(uv)
            valid_total += int(valid.sum())
            accepted_total += int(accepted.sum())
            negative_total += int(neg.sum())
            eps_large_total += int(((EPS / (a * b + EPS) > .01) & accepted).sum())
            a_total += int(accepted.sum())
            cuda_valid_mismatch += int((valid != cuda_valid).sum())
            # Independent finite difference of the double-precision warp at fixed
            # masks. No geometry predicate or acceptance mask is differentiated.
            if finite_difference_report is None and int(accepted.sum()):
                with torch.no_grad():
                    def at(delta):
                        xx, yy, _ = warped_patches(d.detach() + delta, n.detach(), uv, rotation.double(), translation.double(),
                                                   cam.gray_image.squeeze().double(), near.gray_image.squeeze().double(), intr_r, intr_n)
                        return score(xx, yy)
                    # Exclude near-zero exact derivatives from relative error summaries.
                    measurable = accepted & (exact_d.abs() > 1e-3)
                    fd_steps = []
                    for step in (1e-5, 1e-7, 1e-8):
                        fd = (at(step) - at(-step)) / (2 * step)
                        fd_steps.append({
                            "depth_step": step,
                            "abs_error": quantiles((fd - exact_d).abs()[measurable].cpu()),
                            "relative_error": quantiles(((fd - exact_d).abs() / exact_d.abs().clamp_min(1e-8))[measurable].cpu()),
                        })
                    finite_difference_report = {
                        "n_measurable": int(measurable.sum()),
                        "reference_fd_step_sweep": fd_steps,
                        "cuda_vs_exact_abs_error": quantiles((cuda_gd.double() - exact_d).abs()[measurable].cpu()),
                        "cuda_vs_legacy_abs_error": quantiles((cuda_gd.double() - legacy_d).abs()[measurable].cpu()),
                    }
            row = {"reference": cam.image_name, "nearest": near.image_name, "sampled": len(uv), "valid": int(valid.sum()),
                   "accepted": int(accepted.sum()), "negative_accepted": int(neg.sum())}
            all_rows.append(row)
            print(json.dumps(row), flush=True)
    return {
        "model": str(model_path), "scan": model_path.name, "iteration": scene.loaded_iter,
        "seed": args.seed, "sampling": "evenly spaced reference cameras; seeded random nearest; seeded random d_mask pixels",
        "views_requested": args.views, "samples_per_view_max": args.samples,
        "n_sampled": sampled_total, "n_valid": valid_total, "n_accepted_tau_0_9": accepted_total,
        "n_negative_accepted": negative_total,
        "fraction_negative_among_accepted": negative_total / accepted_total if accepted_total else None,
        "n_cuda_accepted_tau_0_9": cuda_accepted_total,
        "n_cuda_negative_accepted": cuda_negative_total,
        "fraction_negative_among_cuda_accepted": cuda_negative_total / cuda_accepted_total if cuda_accepted_total else None,
        "n_cuda_valid_score_over_one": cuda_score_over_one,
        "exact_backend_compared": args.exact_backend,
        "fraction_eps_denominator_over_1pct_among_accepted": eps_large_total / a_total if a_total else None,
        "n_cuda_reference_validity_mismatches": cuda_valid_mismatch,
        "distributions": {k: quantiles(v) for k, v in arrays.items()},
        "finite_difference_first_view": finite_difference_report, "views": all_rows,
        "limitations": ["Frozen final geometry, not the full training trajectory.",
                        "Negative correlation is not ground-truth correspondence error.",
                        "No mesh, novel-view-quality, or runtime improvement is claimed.",
                        "Float32 cancellation and bilinear interpolation kinks can affect CUDA/reference comparisons."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--views", type=int, default=6)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact_backend", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.views < 1 or args.samples < 1:
        parser.error("views and samples must be positive")
    result = {"synthetic": synthetic_checks(), "torch": torch.__version__, "epsilon": EPS}
    if args.model:
        result["real_data"] = audit_model(args)
    source = ROOT / "submodules/warp-patch-ncc/cuda_warp_patch_ncc/warp_patch_ncc_impl.cu"
    result["kernel_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(json.dumps(result["synthetic"], indent=2))
    if "real_data" in result:
        print(json.dumps({k: v for k, v in result["real_data"].items() if k != "views"}, indent=2))


if __name__ == "__main__":
    main()
