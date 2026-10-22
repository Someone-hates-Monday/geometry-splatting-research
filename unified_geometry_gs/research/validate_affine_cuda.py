"""Compare saved CPU ray diagnostics with the installed GGGS CUDA rasterizer.

Run after paired training exits to avoid GPU contention. This checks sampled
forward outputs, not every Gaussian's preprocessing or the backward pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def quantiles(values):
    if not values:
        return None
    return dict(zip(("min", "p50", "p90", "p99", "max"),
                    map(float, np.quantile(values, [0, .5, .9, .99, 1]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth_atol", type=float, default=1e-4)
    parser.add_argument("--alpha_atol", type=float, default=1e-3)
    args = parser.parse_args()
    if args.out.exists() or min(args.depth_atol, args.alpha_atol) <= 0:
        parser.error("Output must be new and absolute tolerances positive.")
    reference = json.loads(args.cpu.read_text())
    model_path = Path(reference["model"])
    ply = model_path / "point_cloud" / f"iteration_{reference['iteration']}" / "point_cloud.ply"
    ply_hash = hashlib.sha256(ply.read_bytes()).hexdigest()
    recorded_hashes = [v for k, v in reference["source_sha256"].items() if k.endswith("point_cloud.ply")]
    if recorded_hashes != [ply_hash]:
        raise ValueError("CPU diagnostic and CUDA validation must use the same frozen PLY.")

    from arguments import ModelParams, PipelineParams, get_combined_args
    import diff_gaussian_rasterization
    from gaussian_renderer import render
    from scene import GaussianModel, Scene

    previous_argv = sys.argv
    try:
        sys.argv = ["validate_affine_cuda.py", "-m", str(model_path)]
        cfg_parser = argparse.ArgumentParser()
        model = ModelParams(cfg_parser, sentinel=True)
        pipeline = PipelineParams(cfg_parser)
        combined = get_combined_args(cfg_parser)
        dataset, pipe = model.extract(combined), pipeline.extract(combined)
    finally:
        sys.argv = previous_argv

    by_camera = {}
    for row in reference["rows"]:
        by_camera.setdefault(row["camera"], []).append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="affine-scene-", dir=args.out.parent.resolve()) as temporary:
        scratch = Path(temporary)
        (scratch / "point_cloud").symlink_to(model_path / "point_cloud", target_is_directory=True)
        dataset.model_path = str(scratch)
        gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
        scene = Scene(dataset, gaussians, load_iteration=reference["iteration"], shuffle=False)
        cameras = {cam.image_name: cam for cam in scene.getTrainCameras() + scene.getTestCameras()}
        bg = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device="cuda")
        with torch.no_grad():
            for name, references in by_camera.items():
                cam = cameras[name]
                output = render(cam, gaussians, pipe, bg, dataset.kernel_size, require_depth=True)
                depth = output["median_depth"].squeeze().cpu().numpy()
                alpha = output["mask"].squeeze().cpu().numpy()
                slope = output["interface_slope"].squeeze().cpu().numpy()
                for ref in references:
                    x, y = ref["pixel"]
                    cuda_depth = float(depth[y, x])
                    ray_length = float(np.sqrt(1 + ((x - cam.Cx) / cam.Fx)**2 + ((y - cam.Cy) / cam.Fy)**2))
                    expected_valid = bool(ref["finite_root"] and ref["reference_fixed_bracket_ok"]
                                          and ref["renderer_final_T_gate"] and not ref["negative_camera_depth"])
                    row = {"camera": name, "pixel": [x, y], "cpu_expected_valid": expected_valid,
                           "cuda_valid": cuda_depth > 0, "cuda_camera_depth": cuda_depth,
                           "alpha_absolute_error": abs((1 - ref["final_transmittance"]) - float(alpha[y, x]))}
                    if ref["finite_root"]:
                        row["cpu_camera_depth"] = ref["median_ray_depth"] / ray_length
                    if expected_valid and cuda_depth > 0:
                        row["depth_absolute_error"] = abs(row["cpu_camera_depth"] - cuda_depth)
                        row["slope_absolute_error"] = abs(ref["transmittance_interface_slope"] - float(slope[y, x]))
                        row["slope_relative_error"] = row["slope_absolute_error"] / max(ref["transmittance_interface_slope"], 1e-12)
                    rows.append(row)
                print(f"CUDA parity {name}: {len(references)} rays", flush=True)
    common = [r for r in rows if "depth_absolute_error" in r]
    mask_mismatches = sum(r["cpu_expected_valid"] != r["cuda_valid"] for r in rows)
    alpha_failures = sum(r["alpha_absolute_error"] > args.alpha_atol for r in rows)
    depth_failures = sum(r["depth_absolute_error"] > args.depth_atol for r in common)
    binary = Path(diff_gaussian_rasterization._C.__file__)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "cpu_input": str(args.cpu.resolve()),
        "model": str(model_path), "sampled_rays": len(rows), "common_valid_rays": len(common),
        "tolerances": {"depth_absolute_normalized_scene_units": args.depth_atol, "alpha_absolute": args.alpha_atol},
        "validity_mismatches": mask_mismatches, "depth_over_tolerance": depth_failures,
        "alpha_over_tolerance": alpha_failures,
        "sampled_forward_parity_pass": bool(common and not (mask_mismatches or depth_failures or alpha_failures)),
        "summary": {key: quantiles([r[key] for r in rows if key in r]) for key in
                    ("depth_absolute_error", "alpha_absolute_error", "slope_absolute_error", "slope_relative_error")},
        "source_sha256": {"point_cloud.ply": ply_hash, "cpu_input": hashlib.sha256(args.cpu.read_bytes()).hexdigest(),
                          "validation_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          str(binary): hashlib.sha256(binary.read_bytes()).hexdigest()},
        "rows": rows,
        "limitations": ["Sampled forward comparison; not per-contributor or backward equivalence.",
            "CPU float64 roots and CUDA float32 fixed-interval interpolation differ numerically.",
            "Slope is a CUDA finite-bracket secant; its discrepancy is reported, not tested for equality.",
            "Passing parity does not establish ground-truth quality or a causal mechanism."],
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
