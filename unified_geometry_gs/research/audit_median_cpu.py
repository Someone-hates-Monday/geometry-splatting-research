"""CPU-only frozen-model diagnostics for continuous affine GGGS rays.

Samples image pixels uniformly, not conditionally on NCC/ground-truth foreground.
The CPU projection is pending parity validation against the installed rasterizer.
Results are exploratory reference-model measurements, not production failure rates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.affine_rays_cpu import read_model, project_model, ray_parameters
from research.median_solver import log_fields, solve


def quantiles(values):
    if not values:
        return None
    return dict(zip(("min", "p50", "p90", "p99", "max"), map(float, np.quantile(values, [0, .5, .9, .99, 1]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--rays", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or min(args.views, args.rays) <= 0:
        parser.error("Output must be new; views and rays must be positive.")
    start = time.perf_counter()
    path = args.model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    model = read_model(path)
    cameras = json.loads((args.model / "cameras.json").read_text())
    # Existing DTU pilot recipe. Refuse silent use on another projection configuration.
    config = (args.model / "cfg_args").read_text()
    if "resolution=2," not in config or "kernel_size=0.0," not in config:
        raise ValueError("This diagnostic currently supports only audited resolution=2, kernel_size=0.")
    camera_ids = np.linspace(0, len(cameras) - 1, min(args.views, len(cameras)), dtype=int)
    rng = np.random.default_rng(args.seed)
    rows = []
    for camera_id in camera_ids:
        camera = cameras[int(camera_id)]
        projected = project_model(model, camera)
        pixels = np.column_stack((rng.integers(0, projected["image_width"], args.rays),
                                  rng.integers(0, projected["image_height"], args.rays)))
        for pixel in pixels:
            a, p, w, init = ray_parameters(projected, pixel)
            total = float((-np.log1p(-a)).sum())
            row = {"camera": camera["img_name"], "pixel": pixel.tolist(), "contributors": len(a),
                   "final_transmittance": float(np.exp(-total)), "finite_root": bool(total > np.log(2))}
            if row["finite_root"]:
                root = solve(a, p, w, tolerance=1e-9)
                depth = float(root["depth"])
                lt, lv, slope = log_fields(depth, a, p, w)
                lower, upper = max(init - .4, 0), max(init + .4, 0)
                legacy_bracket_ok = (log_fields(lower, a, p, w)[0] >= -np.log(2)
                                     and log_fields(upper, a, p, w)[0] <= -np.log(2))
                row.update(median_ray_depth=depth, log_vacancy_transmittance_gap=lv - lt,
                           vacancy_at_median=float(np.exp(lv)), log_interface_slope=slope,
                           transmittance_interface_slope=float(np.exp(lt) * slope),
                           dimensionless_log_slope=float(slope * np.median(w)),
                           root_absolute_error_bound=float(root["absolute_error_bound"]),
                           reference_fixed_bracket_ok=bool(legacy_bracket_ok),
                           renderer_final_T_gate=float(np.exp(-total)) <= .45,
                           negative_camera_depth=depth < 0,
                           post_peak_weight_fraction=float((-np.log1p(-a[p < depth])).sum() / total))
            rows.append(row)
        print(f"CPU audit {camera['img_name']}: {args.rays} rays", flush=True)
        del projected
    valid = [r for r in rows if r["finite_root"]]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "model": str(args.model.resolve()),
        "iteration": args.iteration, "gaussians": len(model[0]), "seed": args.seed,
        "sampled_rays": len(rows), "finite_root_rays": len(valid),
        "summary": {key: quantiles([r[key] for r in valid]) for key in
                    ("log_vacancy_transmittance_gap", "vacancy_at_median", "transmittance_interface_slope", "dimensionless_log_slope")},
        "gap_over_0_01_rays": sum(r["log_vacancy_transmittance_gap"] > .01 for r in valid),
        "fixed_bracket_failure_rays": sum(not r["reference_fixed_bracket_ok"] for r in valid),
        "finite_root_but_final_T_gate_rejects": sum(not r["renderer_final_T_gate"] for r in valid),
        "rows": rows, "wall_seconds": time.perf_counter() - start,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (Path(__file__), Path(__file__).with_name("affine_rays_cpu.py"), path)},
        "limitations": ["Float64 CPU affine transcription; not yet validated against installed CUDA output.",
            "Tile boundaries, float32 sums and early-stop contributor lists may differ from CUDA.",
            "Per-ray affine vacancy need not define one common spatial scalar field across views.",
            "Uniform image rays include background; finite roots are a selected subset.",
            "No ground-truth geometry or causal explanation of reconstruction errors is established.",
            "Intervals and depth are in normalized scene ray-distance units, not DTU millimeters."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
