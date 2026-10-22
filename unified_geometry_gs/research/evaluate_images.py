"""Read-only frozen-model image evaluation with explicit cameras and conventions.

Run after training to avoid contaminating training runtime with GPU contention.
The saved dataset split is honored. A model trained with eval=False has no novel
test split; this script never relabels its training cameras as novel views.
"""
from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every camera; positive values are smoke checks.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Refusing to overwrite an existing evaluation.")
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    original_argv = sys.argv
    model_path = args.model.resolve()
    try:
        sys.argv = ["evaluate_images.py", "-m", str(model_path)]
        cfg_parser = argparse.ArgumentParser()
        model = ModelParams(cfg_parser, sentinel=True)
        pipeline = PipelineParams(cfg_parser)
        combined = get_combined_args(cfg_parser)
        dataset, pipe = model.extract(combined), pipeline.extract(combined)
    finally:
        sys.argv = original_argv
    if args.split == "test" and not dataset.eval:
        parser.error("Saved model used eval=False: there is no independent test-camera split.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="image-scene-", dir=args.out.parent.resolve()) as temporary:
        scratch = Path(temporary)
        (scratch / "point_cloud").symlink_to(model_path / "point_cloud", target_is_directory=True)
        dataset.model_path = str(scratch)
        gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        cameras = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
        if not cameras:
            raise RuntimeError("Selected split contains no cameras.")
        if args.limit > 0:
            cameras = cameras[:args.limit]
        bg = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device="cuda")
        rows = []
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for cam in cameras:
                image = render(cam, gaussians, pipe, bg, dataset.kernel_size, require_depth=False)["render"].clamp(0, 1)
                gt = cam.original_image.cuda().clamp(0, 1)
                mse = (image - gt).square().mean()
                rows.append({"camera": cam.image_name, "height": cam.image_height, "width": cam.image_width,
                    "psnr_global_rgb_db": float((-10 * torch.log10(mse.clamp_min(1e-20))).item()),
                    "psnr_repository_channel_mean_db": float(psnr(image, gt).mean().item()),
                    "ssim_repository": float(ssim(image.unsqueeze(0), gt.unsqueeze(0)).item())})
        ply = model_path / "point_cloud" / f"iteration_{scene.loaded_iter}" / "point_cloud.ply"
        checkpoint = model_path / f"chkpnt{scene.loaded_iter}.pth"
        result = {
            "model": str(model_path), "iteration": scene.loaded_iter, "split": args.split,
            "is_novel_view_evaluation": args.split == "test", "partial_smoke": args.limit > 0,
            "camera_count": len(rows), "rows": rows,
            "mean": {key: float(np.mean([row[key] for row in rows])) for key in
                     ("psnr_global_rgb_db", "psnr_repository_channel_mean_db", "ssim_repository")},
            "gaussian_count": len(gaussians.get_xyz), "gaussian_ply_bytes": ply.stat().st_size,
            "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
            "background": "white" if dataset.white_background else "black",
            "resolution_parameter": dataset.resolution, "appearance": "Raw renderer, matching train.py report; no test-time exposure fitting.",
            "wall_seconds_including_load": time.perf_counter() - started,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "source_sha256": {"point_cloud.ply": hashlib.sha256(ply.read_bytes()).hexdigest(),
                              "cfg_args": hashlib.sha256((model_path / "cfg_args").read_bytes()).hexdigest(),
                              "evaluate_images.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "limitations": ["LPIPS and rendering-speed benchmark are not included.",
                            "Training-view quality does not establish novel-view generalization.",
                            "Global RGB MSE PSNR and repository per-channel mean PSNR are different conventions."],
        }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("camera_count", "split", "mean", "gaussian_count")}, indent=2))


if __name__ == "__main__":
    main()
