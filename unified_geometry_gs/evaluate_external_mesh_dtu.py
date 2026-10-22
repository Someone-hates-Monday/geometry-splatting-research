#!/usr/bin/env python3
"""
Align an external reconstruction mesh (e.g. SuGaR export) to the DTU evaluation frame
using the same Sim(3)-style alignment as evaluate_dtu_mesh.py (train camera centers vs
DTU calibration poses), then run dtu_eval/eval.py.

Does not load Gaussian checkpoints or cfg_args — only COLMAP extrinsics under --source_path.

Writes:
  --out_dir/recon_aligned.ply
  --out_dir/vis/results.json

Optional:
  Copy input mesh to --out_dir/recon_post.ply for orbit snapshots (same convention as UGGS).

Example:
  python evaluate_external_mesh_dtu.py \\
    --source_path \"$UGGS_DTU_ROOT/scan24\" \\
    --mesh_in /path/to/sugar_mesh.ply \\
    --out_dir ./output_compare/sugar/scan24 \\
    --scan_id 24 \\
    --DTU ./dtu_eval/Offical_DTU_Dataset
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluate_dtu_mesh import best_fit_transform, load_dtu_camera
from scene.dataset_readers import readColmapSceneInfo
from utils.graphics_utils import getWorld2View2


def train_camera_centers_from_colmap(source_path: str, images_subdir: str) -> np.ndarray:
    scene_info = readColmapSceneInfo(source_path, images_subdir, eval=False)
    pts: list[np.ndarray] = []
    for cam in scene_info.train_cameras:
        w2c = getWorld2View2(cam.R, cam.T)
        c2w = np.linalg.inv(w2c)
        pts.append(c2w[:3, 3].astype(np.float64))
    return np.stack(pts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="DTU mesh eval for external/aligned-free folders.")
    parser.add_argument("--source_path", type=str, required=True, help="DTU scan COLMAP root (contains sparse/0, images/).")
    parser.add_argument("--mesh_in", type=str, required=True, help="Input mesh ply (same scene frame as COLMAP).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder (vis/results.json, recon_aligned.ply).")
    parser.add_argument("--scan_id", type=int, required=True, help="DTU scan id (e.g. 24 for scan24).")
    parser.add_argument("--DTU", type=str, default="dtu_eval/Offical_DTU_Dataset")
    parser.add_argument(
        "--images",
        type=str,
        default="images",
        help="Images subdirectory under source_path (COLMAP undistorted images).",
    )
    parser.add_argument(
        "--write_recon_post",
        action="store_true",
        help="Copy mesh_in to out_dir/recon_post.ply (for capture_mesh_snapshots convention).",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Only align and write recon_aligned.ply; skip dtu_eval/eval.py.",
    )
    args = parser.parse_args()

    source_path = os.path.abspath(args.source_path)
    mesh_in = Path(args.mesh_in).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    dtu_path = os.path.abspath(args.DTU)
    cwd = str(_SCRIPT_DIR)

    if not mesh_in.is_file():
        print(f"Missing mesh: {mesh_in}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = train_camera_centers_from_colmap(source_path, args.images)
    dtu_cameras = load_dtu_camera(dtu_path)
    gt_points = np.array([cam[:, 3] for cam in dtu_cameras], dtype=np.float64)
    gt_points = gt_points[: points.shape[0]]

    scale_points = np.linalg.norm(points - points.mean(axis=0), axis=1).mean()
    scale_gt_points = np.linalg.norm(gt_points - gt_points.mean(axis=0), axis=1).mean()
    if scale_points < 1e-12:
        print("Degenerate train camera spread; cannot scale.", file=sys.stderr)
        sys.exit(1)

    scaled_pts = points * (scale_gt_points / scale_points)
    _, r, t = best_fit_transform(scaled_pts, gt_points)

    mesh = trimesh.load(str(mesh_in))
    mesh.vertices = mesh.vertices * (scale_gt_points / scale_points)
    mesh.vertices = mesh.vertices @ r.T + t

    aligned_path = out_dir / "recon_aligned.ply"
    mesh.export(str(aligned_path))
    print(f"Wrote {aligned_path}")

    if args.write_recon_post:
        post_path = out_dir / "recon_post.ply"
        shutil.copy2(mesh_in, post_path)
        print(f"Wrote {post_path} (copy of input mesh)")

    if args.skip_eval:
        return

    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    eval_py = _SCRIPT_DIR / "dtu_eval" / "eval.py"
    cmd = [
        sys.executable,
        str(eval_py),
        "--data",
        str(aligned_path),
        "--scan",
        str(int(args.scan_id)),
        "--mode",
        "mesh",
        "--dataset_dir",
        dtu_path,
        "--vis_out_dir",
        str(vis_dir),
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    results_json = vis_dir / "results.json"
    if not results_json.is_file():
        print(f"Expected {results_json} after eval.", file=sys.stderr)
        sys.exit(1)
    print(results_json.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
