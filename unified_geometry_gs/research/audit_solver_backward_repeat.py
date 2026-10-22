"""Compare actual rasterizer depth gradients on fixed frozen-model pixels.

Checks finite gradients and drift relative to the same-source five-pass control;
not an independent derivative oracle and not a training-quality experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.benchmark_median_cuda import load_config
from research.median_cuda_backend import activate, load_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", nargs="+", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Fresh output required.")
    started = time.monotonic()
    while True:
        state = json.loads(args.after.read_text())
        if state["status"] == "complete":
            break
        if state["status"] == "failed":
            raise RuntimeError("Prior mesh experiment failed; inspect it before continuing.")
        if time.monotonic()-started > 3600:
            raise TimeoutError("Previous experiment did not finish.")
        time.sleep(10)
    from scene import GaussianModel, Scene
    from gaussian_renderer import render
    backends = {"control": load_backend("control"), "control_repeat": load_backend("control"), "four": load_backend("four")}
    fields = ("_xyz", "_scaling", "_rotation", "_opacity")
    all_fields = fields + ("_features_dc", "_features_rest", "_sg_axis", "_sg_sharpness", "_sg_color")
    rows, sources = [], {}
    rng = np.random.default_rng(0)
    for cpu_path in args.cpu:
        ref = json.loads(cpu_path.read_text())
        model_path = Path(ref["model"])
        ply = model_path / "point_cloud/iteration_30000/point_cloud.ply"
        ply_hash = hashlib.sha256(ply.read_bytes()).hexdigest()
        if [v for k,v in ref["source_sha256"].items() if k.endswith("point_cloud.ply")] != [ply_hash]:
            raise ValueError("Frozen model hash changed.")
        sources[str(ply)] = ply_hash
        names = list(dict.fromkeys(r["camera"] for r in ref["rows"]))[::4]
        dataset, pipe = load_config(model_path)
        with tempfile.TemporaryDirectory(prefix="backward-scene-", dir=args.out.parent.resolve()) as temp:
            scratch = Path(temp)
            (scratch / "point_cloud").symlink_to(model_path / "point_cloud", target_is_directory=True)
            dataset.model_path = str(scratch)
            gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
            scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
            cameras = {c.image_name: c for c in scene.getTrainCameras() + scene.getTestCameras()}
            bg = torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.], device="cuda")
            for name in names:
                cam = cameras[name]
                selected = [r for r in ref["rows"] if r["camera"] == name and r["finite_root"] and
                    r["reference_fixed_bracket_ok"] and r["renderer_final_T_gate"] and not r["negative_camera_depth"]]
                pixels = np.asarray([r["pixel"] for r in selected])
                if not len(pixels):
                    raise ValueError("No valid sampled pixels.")
                xy = torch.as_tensor(pixels, dtype=torch.long, device="cuda")
                signs = rng.choice([-1.,1.], size=len(pixels))
                weights = torch.as_tensor(signs, device="cuda", dtype=torch.float32)
                gradients, losses = {}, {}
                for variant, backend in backends.items():
                    for field in all_fields:
                        getattr(gaussians, field).grad = None
                    with activate(backend):
                        output = render(cam, gaussians, pipe, bg, dataset.kernel_size, require_depth=True)
                        depth = output["median_depth"].squeeze()
                        if not bool((depth[xy[:,1],xy[:,0]] > 0).all()):
                            raise ValueError("Selected depth mask changed.")
                        loss = (depth[xy[:,1],xy[:,0]] * weights).mean()
                        loss.backward()
                        torch.cuda.synchronize()
                    losses[variant] = float(loss.detach())
                    gradients[variant] = {f: getattr(gaussians,f).grad.detach().cpu().double() for f in fields}
                    del output, depth, loss
                comparisons = {}
                for field in fields:
                    a, b = (gradients[v][field] for v in ("control", "four"))
                    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                    error = b-a
                    repeat_error = gradients["control_repeat"][field]-a
                    norm_a = float(torch.linalg.vector_norm(a))
                    comparisons[field] = {"all_finite": finite, "control_l2": norm_a,
                        "four_l2": float(torch.linalg.vector_norm(b)),
                        "relative_l2_drift": float(torch.linalg.vector_norm(error))/max(norm_a,1e-12),
                        "control_repeat_relative_l2_drift": float(torch.linalg.vector_norm(repeat_error))/max(norm_a,1e-12),
                        "max_absolute_difference": float(error.abs().max()),
                        "parameter_elements": a.numel()}
                rows.append({"model": str(model_path), "camera": name, "pixels": pixels.tolist(),
                    "rademacher_weights": signs.tolist(), "losses": losses, "gradient_comparison": comparisons})
                print(f"Depth backward {model_path.name}/{name}: " +
                    ", ".join(f"{f} drift={c['relative_l2_drift']:.3g}" for f,c in comparisons.items()), flush=True)
                del gradients
            del gaussians, scene, cameras
            torch.cuda.empty_cache()
    for p in (Path(__file__), ROOT / "research/median_cuda_backend.py", ROOT / "research/benchmark_median_cuda.py"):
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    for backend in backends.values():
        p = Path(backend.__file__)
        sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "rows": rows,
        "all_gradients_finite": all(c["all_finite"] for r in rows for c in r["gradient_comparison"].values()),
        "relative_l2_drift_engineering_threshold": .001,
        "all_groups_drift_below_threshold": all(c["relative_l2_drift"] <= .001 for r in rows for c in r["gradient_comparison"].values()),
        "source_sha256": sources,
        "limitations": ["Same backward source evaluated at different approximate depth roots; not an independent oracle.",
            "Six fixed cameras and their sampled valid pixels, not all cameras/pixels or training trajectories.",
            "Threshold 0.001 is an engineering drift rule, not a statistical or derivative-correctness guarantee.",
            "Only Gaussian xyz/scaling/rotation/opacity gradients are compared; no appearance training test."]}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("all_gradients_finite", "all_groups_drift_below_threshold")}), flush=True)


if __name__ == "__main__":
    main()
