"""Count solver work on the exact ray parameters of saved CPU diagnostics.

Independent affine model remains pending CUDA parity. Function counts are a
mechanism diagnostic; they omit sorting, derivative cost and GPU synchronization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.affine_rays_cpu import read_model, project_model, ray_parameters
from research.median_newton import solve_newton
from research.median_solver import solve


def quantiles(values):
    return dict(zip(("min", "p50", "p90", "p99", "max"),
                    map(float, np.quantile(values, [0, .5, .9, .99, 1]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    if args.out.exists() or args.tolerance <= 0:
        parser.error("Output must be new and tolerance positive.")
    summaries = []
    for path in args.cpu:
        reference = json.loads(path.read_text())
        model_path = Path(reference["model"])
        ply = model_path / "point_cloud" / f"iteration_{reference['iteration']}" / "point_cloud.ply"
        recorded = [v for k, v in reference["source_sha256"].items() if k.endswith("point_cloud.ply")]
        if recorded != [hashlib.sha256(ply.read_bytes()).hexdigest()]:
            raise ValueError("Frozen PLY changed after the CPU diagnostic.")
        model = read_model(ply)
        cameras = {c["img_name"]: c for c in json.loads((model_path / "cameras.json").read_text())}
        rows = []
        grouped = {}
        for row in reference["rows"]:
            if row["finite_root"]:
                grouped.setdefault(row["camera"], []).append(row)
        for name, references in grouped.items():
            projected = project_model(model, cameras[name])
            for ref in references:
                a, p, w, initial = ray_parameters(projected, ref["pixel"])
                new_mid = solve_newton(a, p, w, tolerance=args.tolerance)
                new_init = solve_newton(a, p, w, tolerance=args.tolerance, initial_depth=initial)
                old = solve(a, p, w, tolerance=args.tolerance)
                bisect = solve(a, p, w, tolerance=args.tolerance, newton=False)
                for result in (new_mid, new_init):
                    if abs(result["depth"] - ref["median_ray_depth"]) > result["absolute_error_bound"] + ref["root_absolute_error_bound"] + 1e-10:
                        raise ArithmeticError(f"Root discrepancy at {name}:{ref['pixel']}")
                rows.append({"camera": name, "pixel": ref["pixel"], "contributors": len(a),
                    "cpu_expected_renderer_valid": bool(ref["reference_fixed_bracket_ok"] and ref["renderer_final_T_gate"] and not ref["negative_camera_depth"]),
                    "new_mid_evaluations": new_mid["log_transmittance_evaluations"],
                    "new_init_evaluations": new_init["log_transmittance_evaluations"],
                    "new_mid_high_precision_passes": new_mid["high_precision_evaluations"],
                    "new_init_high_precision_passes": new_init["high_precision_evaluations"],
                    "old_hybrid_evaluations": old["log_transmittance_evaluations"],
                    "bisection_evaluations": bisect["log_transmittance_evaluations"],
                    "new_init_depth_error": abs(new_init["depth"] - ref["median_ray_depth"])})
            print(f"Solver audit {model_path.name}/{name}: {len(references)} finite rays", flush=True)
            del projected
        summaries.append({"model": str(model_path), "cpu_input": str(path.resolve()), "finite_rays": len(rows),
            "cpu_expected_renderer_valid_rays": sum(r["cpu_expected_renderer_valid"] for r in rows),
            "summary": {key: quantiles([r[key] for r in rows]) for key in
                ("contributors", "new_mid_evaluations", "new_init_evaluations", "old_hybrid_evaluations", "bisection_evaluations", "new_init_depth_error")},
            "high_precision_passes": {key: sum(r[key] for r in rows) for key in ("new_mid_high_precision_passes", "new_init_high_precision_passes")},
            "cpu_input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": rows})
    scripts = [Path(__file__), Path(__file__).with_name("median_newton.py"), Path(__file__).with_name("median_solver.py"), Path(__file__).with_name("affine_rays_cpu.py")]
    result = {"tolerance_ray_scene_units": args.tolerance, "scenes": summaries,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
        "limitations": ["CPU affine transcription pending CUDA validation.",
            "Function counts omit sorting, derivative overhead, high-precision cost and GPU synchronization.",
            "Different initialization policies are reported separately; no GPU acceleration or quality claim.",
            "Endpoint checks are ordinary or 80-digit numerical evaluations, not formal interval arithmetic."]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps([{k: v for k, v in s.items() if k != "rows"} for s in summaries], indent=2))


if __name__ == "__main__":
    main()
