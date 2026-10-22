"""Frozen-model five-versus-four subdivision extraction and seeded mesh evaluation.

Requires completed GPU benchmark first. Both arms regenerate meshes from the
same PLY in fresh scratch model folders. No training or historical files change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.benchmark_median_cuda import load_config
from research.median_cuda_backend import activate, load_backend


def dump(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), ROOT / "research/median_cuda_backend.py",
        ROOT / "research/benchmark_median_cuda.py", ROOT / "research/evaluate_mesh_seeded.py",
        ROOT / "mesh_extract.py", ROOT / "evaluate_dtu_mesh.py", ROOT / "dtu_eval/eval.py"]
    state = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "waiting_for_benchmark",
        "steps": [], "voxel_size": .002, "trunc_voxel_multiplier": 4., "depth_max": 8.,
        "clusters": 1, "evaluation_seed": 0,
        "prespecified_quality_tolerance_mm": .0005,
        "quality_rule": "Absolute overall delta <= tolerance and increases in either directed distance <= tolerance, in every scene.",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    manifest = args.out / "manifest.json"
    dump(manifest, state)
    started_wait = time.monotonic()
    try:
        while not args.after.is_file():
            if time.monotonic() - started_wait > 3600:
                raise TimeoutError("Benchmark result not produced within one hour.")
            time.sleep(10)
        benchmark = json.loads(args.after.read_text())
        controls = [r for r in benchmark["rows"] if r["variant"] == "control"]
        four = [r for r in benchmark["rows"] if r["variant"] == "four"]
        if not controls or not four or any(r["full_image_validity_mismatches"] or
            r["sampled_depth_errors_gt_1e_4"] for r in controls + four):
            raise ValueError("Control/four frozen-render correctness gate failed.")
        for path, digest in state["source_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Queued source changed: {path}")
        state["status"] = "running"
        dump(manifest, state)
        from mesh_extract import extract_mesh
        from scene import GaussianModel, Scene
        from evaluate_dtu_mesh import best_fit_transform, load_dtu_camera
        backends = {"five": load_backend("control"), "four": load_backend("four")}
        gt_poses = load_dtu_camera(str(ROOT / "dtu_eval/Offical_DTU_Dataset"))
        gt_centers = np.asarray([p[:, 3] for p in gt_poses])
        paired = []
        for model_path in args.models:
            model_path = model_path.resolve()
            scan = int(model_path.name.removeprefix("scan"))
            ply = model_path / "point_cloud/iteration_30000/point_cloud.ply"
            state["source_sha256"][str(ply)] = hashlib.sha256(ply.read_bytes()).hexdigest()
            arm_metrics = {}
            for arm, backend in backends.items():
                folder = args.out / model_path.name / arm
                folder.mkdir(parents=True, exist_ok=False)
                (folder / "point_cloud").symlink_to(model_path / "point_cloud", target_is_directory=True)
                dataset, pipe = load_config(model_path)
                dataset.model_path = str(folder)
                step = {"scan": scan, "arm": arm, "status": "extracting"}
                state["steps"].append(step)
                dump(manifest, state)
                started = time.perf_counter()
                with torch.no_grad(), activate(backend):
                    extract_mesh(dataset, pipe, 30000, num_cluster=1, voxel_size=.002,
                        depth_max=8., trunc_voxel_multiplier=4., write_dir=str(folder))
                step["extract_wall_seconds_including_load"] = time.perf_counter()-started
                # Same camera-center similarity transform as evaluate_dtu_mesh,
                # without its os.system call to an unseeded evaluator.
                gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
                scene = Scene(dataset, gaussians, load_iteration=30000, shuffle=False)
                centers = np.asarray([(c.world_view_transform.T).inverse()[:3, 3].cpu().numpy() for c in scene.getTrainCameras()])
                target = gt_centers[:len(centers)]
                scale = np.linalg.norm(target-target.mean(0), axis=1).mean() / np.linalg.norm(centers-centers.mean(0), axis=1).mean()
                _, rotation, translation = best_fit_transform(centers*scale, target)
                mesh = trimesh.load(folder / "recon_post.ply", process=True)
                mesh.vertices = mesh.vertices*scale @ rotation.T + translation
                aligned = folder / "recon_aligned.ply"
                mesh.export(aligned)
                step.update(status="evaluating", mesh_vertices=len(mesh.vertices), mesh_faces=len(mesh.faces),
                    aligned_mesh_sha256=hashlib.sha256(aligned.read_bytes()).hexdigest())
                dump(manifest, state)
                del mesh, gaussians, scene
                torch.cuda.empty_cache()
                command = [sys.executable, str(ROOT / "research/evaluate_mesh_seeded.py"),
                    "--mesh", str(aligned), "--scan", str(scan), "--seed", "0", "--workers", "2",
                    "--out", str(folder / "seed0")]
                step["evaluation_argv"] = command
                with (folder / "seeded_evaluate.log").open("w") as log:
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode:
                    raise RuntimeError(f"Evaluation failed for {scan}/{arm}")
                arm_metrics[arm] = json.loads((folder / "seed0/results.json").read_text())
                step.update(status="complete", metrics=arm_metrics[arm])
                dump(manifest, state)
                print(f"Completed scan{scan}/{arm}: {arm_metrics[arm]}", flush=True)
            delta = {k: arm_metrics["four"][k]-arm_metrics["five"][k] for k in ("overall", "mean_d2s", "mean_s2d")}
            tol = state["prespecified_quality_tolerance_mm"]
            paired.append({"scan": scan, **arm_metrics, "delta_four_minus_five": delta,
                "within_engineering_quality_tolerance": abs(delta["overall"]) <= tol and delta["mean_d2s"] <= tol and delta["mean_s2d"] <= tol})
            dump(args.out / "paired_mesh_metrics.json", {"scenes": paired,
                "all_scenes_within_engineering_tolerance": all(p["within_engineering_quality_tolerance"] for p in paired),
                "complete": len(paired) == len(args.models),
                "limitations": ["Frozen models, one evaluator seed, not training experiments.",
                    "Tolerance is an engineering rule, not a statistical equivalence test.",
                    "Four subdivisions are a simple efficiency control, not a novelty claim."]})
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        dump(manifest, state)


if __name__ == "__main__":
    main()
