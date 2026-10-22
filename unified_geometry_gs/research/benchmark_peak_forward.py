"""Frozen-render correctness and counterbalanced GPU event timing.

Includes an unmodified-source JIT control to distinguish solver changes from
build differences. Measurements are rendering only, not training speedups.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.median_peak_adaptive_backend import activate, load_backend


def quantiles(values):
    return dict(zip(("min", "p50", "p90", "p99", "max"),
        map(float, np.quantile(values, [0, .5, .9, .99, 1])))) if len(values) else None


def load_config(model_path):
    from arguments import ModelParams, PipelineParams, get_combined_args
    previous = sys.argv
    try:
        sys.argv = ["benchmark_median_cuda.py", "-m", str(model_path)]
        parser = argparse.ArgumentParser()
        model, pipeline = ModelParams(parser, sentinel=True), PipelineParams(parser)
        combined = get_combined_args(parser)
        return model.extract(combined), pipeline.extract(combined)
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", nargs="+", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=("control", "four", "adjquad4", "peakadj"), default=["control", "adjquad4", "peakadj"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--views", type=int, default=0, help="0 uses all eight prespecified reference views.")
    args = parser.parse_args()
    if args.out.exists() or min(args.repeats, args.rounds) < 1:
        parser.error("Fresh output and positive repetition counts required.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    import diff_gaussian_rasterization

    backends = {"installed": diff_gaussian_rasterization._C}
    for name in args.variants:
        print(f"Building {name}", flush=True)
        backends[name] = load_backend(name)
    rows = []
    source_hashes = {str(Path(b.__file__)): hashlib.sha256(Path(b.__file__).read_bytes()).hexdigest() for b in backends.values()}
    rng = random.Random(0)
    for cpu_path in args.cpu:
        reference = json.loads(cpu_path.read_text())
        model_path = Path(reference["model"])
        ply = model_path / "point_cloud" / f"iteration_{reference['iteration']}" / "point_cloud.ply"
        ply_hash = hashlib.sha256(ply.read_bytes()).hexdigest()
        if [v for k, v in reference["source_sha256"].items() if k.endswith("point_cloud.ply")] != [ply_hash]:
            raise ValueError("Frozen model hash changed.")
        source_hashes[str(ply)] = ply_hash
        grouped = {}
        for ref in reference["rows"]:
            grouped.setdefault(ref["camera"], []).append(ref)
        names = list(grouped)
        if args.views:
            names = names[:args.views]
        dataset, pipe = load_config(model_path)
        with tempfile.TemporaryDirectory(prefix="solver-scene-", dir=args.out.parent.resolve()) as temp:
            scratch = Path(temp)
            (scratch / "point_cloud").symlink_to(model_path / "point_cloud", target_is_directory=True)
            dataset.model_path = str(scratch)
            gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
            scene = Scene(dataset, gaussians, load_iteration=reference["iteration"], shuffle=False)
            cameras = {c.image_name: c for c in scene.getTrainCameras() + scene.getTestCameras()}
            bg = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device="cuda")
            for name in names:
                cam = cameras[name]
                with torch.no_grad():
                    base = render(cam, gaussians, pipe, bg, dataset.kernel_size, require_depth=True)
                    base_depth = base["median_depth"].squeeze()
                    base_mask = base_depth > 0
                    results = {}
                    for variant, backend in backends.items():
                        with activate(backend):
                            output = render(cam, gaussians, pipe, bg, dataset.kernel_size, require_depth=True)
                        depth = output["median_depth"].squeeze()
                        common = (depth > 0) & base_mask
                        cpu_errors, cpu_mask_mismatches = [], 0
                        for ref in grouped[name]:
                            x, y = ref["pixel"]
                            expected = bool(ref["finite_root"] and ref["reference_fixed_bracket_ok"] and
                                ref["renderer_final_T_gate"] and not ref["negative_camera_depth"])
                            got = float(depth[y, x])
                            cpu_mask_mismatches += (got > 0) != expected
                            if expected and got > 0:
                                ray_length = np.sqrt(1 + ((x-cam.Cx)/cam.Fx)**2 + ((y-cam.Cy)/cam.Fy)**2)
                                cpu_errors.append(abs(got - ref["median_ray_depth"]/ray_length))
                        error = (depth[common]-base_depth[common]).abs().cpu().numpy()
                        results[variant] = {
                            "full_image_validity_mismatches": int(((depth > 0) != base_mask).sum()),
                            "sampled_cpu_validity_mismatches": int(cpu_mask_mismatches),
                            "depth_error_vs_installed": quantiles(error),
                            "sampled_depth_error_vs_cpu": quantiles(cpu_errors),
                            "sampled_depth_errors_gt_1e_4": int(sum(e > 1e-4 for e in cpu_errors)),
                            "rgb_max_abs_difference": float((output["render"]-base["render"]).abs().max()),
                            "alpha_max_abs_difference": float((output["mask"]-base["mask"]).abs().max()),
                            "normal_max_abs_difference": float((output["normal"]-base["normal"]).abs().max()),
                            "gpu_ms_rounds": [], "depth_off_gpu_ms_rounds": [],
                        }
                    del output, base
                    # Warm all variants, then randomize order independently per
                    # round. Host setup and kernel compilation are not timed.
                    for variant, backend in backends.items():
                        with activate(backend):
                            for _ in range(5):
                                render(cam, gaussians, pipe, bg, dataset.kernel_size, require_depth=True)
                    for _ in range(args.rounds):
                        order = [(variant, depth_on) for variant in backends for depth_on in (True, False)]
                        rng.shuffle(order)
                        for variant, depth_on in order:
                            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                            torch.cuda.synchronize()
                            with activate(backends[variant]):
                                start.record()
                                for _ in range(args.repeats):
                                    render(cam, gaussians, pipe, bg, dataset.kernel_size, require_depth=depth_on)
                                end.record()
                                end.synchronize()
                            key = "gpu_ms_rounds" if depth_on else "depth_off_gpu_ms_rounds"
                            results[variant][key].append(start.elapsed_time(end)/args.repeats)
                    for variant, values in results.items():
                        for key in ("gpu_ms_rounds", "depth_off_gpu_ms_rounds"):
                            values[key.replace("_rounds", "_median")] = float(np.median(values[key]))
                        rows.append({"model": str(model_path), "camera": name, "variant": variant, **values})
                    args.out.with_suffix(".partial.json").write_text(json.dumps(
                        {"status": "running", "rows": rows}, indent=2, allow_nan=False) + "\n")
                print(f"Measured {model_path.name}/{name}: " + ", ".join(f"{k}={v['gpu_ms_median']:.3f}ms" for k,v in results.items()), flush=True)
            del gaussians, scene, cameras, base_depth, base_mask
            torch.cuda.empty_cache()
    for path in (Path(__file__), ROOT / "research/median_cuda_backend.py", ROOT / "research/median_hybrid_forward.inc",
                 ROOT / "research/median_peak_adaptive_backend.py", ROOT / "research/median_adjquad_backend.py", ROOT / "research/median_adjquad_accumulate.inc", ROOT / "research/median_adjquad_refine.inc"):
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "rounds": args.rounds, "repeats_per_round": args.repeats,
        "timing_order": "seed-0 randomized variant/depth-on-off order within each camera/round",
        "rows": rows, "source_sha256": source_hashes,
        "limitations": ["Frozen training views, not reconstruction quality, backward validation or training throughput.",
            "CUDA event batches include renderer GPU work and may include CPU launch gaps.",
            "Same-source JIT control can differ from installed binary because of build settings.",
            "Hybrid endpoint margin is an engineering float32 check, not interval arithmetic.",
            "Hybrid accepted slope uses analytic q*H_t; old output uses bracket secant.",
            "Four-subdivision variant is a speed/accuracy control, not a claimed novel method."]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
