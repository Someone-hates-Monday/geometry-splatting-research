"""Fresh, paired legacy/exact-NCC training with immutable experiment metadata.

Usage (WSL uggs environment, from repository root):
  python research/run_ncc_pair.py --scan scan37 --out research/experiments/smoke37 --smoke
  python research/run_ncc_pair.py --scan scan37 --out research/experiments/exact37

Never overwrites an existing directory. Does not remove checkpoints or meshes.
The only scientific difference between full-run arms is --ncc_exact_gradient.
Smoke mode starts regularization early and is exclusively an integration test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", default="scan37")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    source = ROOT.parent / "dtu/DTU" / args.scan
    if not source.is_dir() or not args.scan.startswith("scan"):
        parser.error("Expected an existing DTU scan directory.")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    iterations = 50 if args.smoke else 30000
    reg_start = 1 if args.smoke else 7000
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scan": args.scan, "smoke_only": args.smoke, "seed": 0,
        "seed_policy": "Existing safe_state fixes Python, NumPy and torch to 0; CUDA determinism is not guaranteed.",
        "iterations": iterations, "regularization_start": reg_start,
        "image_evaluation": "Existing train.py final report: sampled training views, not novel-view evaluation.",
        "adoption": "Full run is a pilot. Require multi-scene geometric improvement, PSNR gate and repeated seeds before claims.",
        "source_sha256": {}, "steps": [], "status": "running",
    }
    files = ["train.py", "arguments/__init__.py", "utils/loss_utils.py",
             "research/ncc_exact_backend.py", "research/train_instrumented.py",
             "research/run_ncc_pair.py", "mesh_extract.py", "evaluate_dtu_mesh.py",
             "submodules/warp-patch-ncc/cuda_warp_patch_ncc/warp_patch_ncc_impl.cu"]
    manifest["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}
    import torch
    manifest["environment"] = {"python": sys.version, "torch": torch.__version__,
                               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
    dump(out / "manifest.json", manifest)

    def run(stage, command):
        entry = {"stage": stage, "argv": command, "status": "running"}
        manifest["steps"].append(entry)
        dump(out / "manifest.json", manifest)
        print(f"[{stage}] starting", flush=True)
        start = time.perf_counter()
        with (out / f"{stage}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, wall_seconds=time.perf_counter() - start,
                     status="complete" if result.returncode == 0 else "failed")
        dump(out / "manifest.json", manifest)
        if result.returncode:
            raise RuntimeError(f"{stage} exited {result.returncode}; see {stage}.log")
        print(f"[{stage}] completed in {entry['wall_seconds']:.1f}s", flush=True)

    try:
        for arm in ("legacy", "exact"):
            model = out / arm
            command = [sys.executable, "research/train_instrumented.py", "-s", str(source),
                       "-m", str(model), "-r", "2", "--use_decoupled_appearance", "3",
                       "--iterations", str(iterations), "--regularization_from_iter", str(reg_start),
                       "--save_iterations", str(iterations), "--test_iterations", str(iterations),
                       "--checkpoint_iterations", str(iterations), "--lambda_depth_normal", "0",
                       "--lambda_multi_view_ncc", ".6", "--lambda_multi_view_geo", ".02",
                       "--multi_view_ncc_views", "1", "--ncc_mask_thresh", ".9", "--quiet"]
            if arm == "exact":
                command.append("--ncc_exact_gradient")
            run(f"train_{arm}", command)
            integration = json.loads((model / "training_integration.json").read_text())
            if not integration["completed"] or not integration["calls"]:
                raise RuntimeError(f"{arm} training integration was not exercised.")
            if not args.smoke:
                run(f"extract_{arm}", [sys.executable, "mesh_extract.py", "-m", str(model),
                    "--iteration", str(iterations), "--voxel_size", ".002",
                    "--trunc_voxel_multiplier", "4", "--depth_max", "8", "--num_cluster", "1"])
                run(f"evaluate_{arm}", [sys.executable, "evaluate_dtu_mesh.py", "-m", str(model),
                    "--iteration", str(iterations), "--DTU", str(ROOT / "dtu_eval/Offical_DTU_Dataset")])
                # Legacy evaluator uses os.system, so its exit code alone is insufficient.
                metrics = model / "vis/results.json"
                if not metrics.is_file():
                    raise RuntimeError(f"{arm} mesh evaluator did not produce metrics.")
        if not args.smoke:
            pair = {arm: json.loads((out / arm / "vis/results.json").read_text()) for arm in ("legacy", "exact")}
            pair["delta_exact_minus_legacy"] = {key: pair["exact"][key] - pair["legacy"][key]
                for key in ("overall", "mean_d2s", "mean_s2d")}
            pair["limitation"] = "One scene and one seed; mesh sampling is stochastic. No statistical significance or novel-view image claim."
            dump(out / "paired_mesh_metrics.json", pair)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        dump(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
