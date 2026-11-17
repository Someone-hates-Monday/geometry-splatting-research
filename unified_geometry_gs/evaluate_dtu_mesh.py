# The evalution code is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/evaluate_dtu_mesh.py
import json
import math
import os
import random
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from skimage.morphology import binary_dilation, disk

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.jia_io_guard import DEFAULT_EVAL_ARTIFACTS, assert_fresh_output_dir

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def _find_calibration_dir(DTU):
    """Find Calibration subdir (cal18, cal49, or first cal* with pos_001.txt)."""
    cal_root = os.path.join(DTU, "Calibration")
    if not os.path.isdir(cal_root):
        raise FileNotFoundError(
            f"DTU eval root has no 'Calibration' folder: {cal_root}\n"
            "Required: Official DTU Robot Image Dataset (SampleSet) with Calibration/, ObsMask/, Points/stl/. See INSTALL.md."
        )
    # Prefer cal18 (64 views), then cal49
    for sub in ("cal18", "cal49"):
        p = os.path.join(cal_root, sub, "pos_001.txt")
        if os.path.isfile(p):
            return os.path.join(cal_root, sub)
    for name in sorted(os.listdir(cal_root)):
        sub = os.path.join(cal_root, name)
        if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "pos_001.txt")):
            return sub
    raise FileNotFoundError(
        f"No calibration poses found under {cal_root} (expected cal18/pos_001.txt etc.).\n"
        "Download SampleSet from https://roboimagedata.compute.dtu.dk/?page_id=36 and extract Calibration/."
    )


def load_dtu_camera(DTU):
    # Load projection matrix from file.
    cal_dir = _find_calibration_dir(DTU)
    camtoworlds = []
    for n in (64, 49):
        last = os.path.join(cal_dir, f"pos_{n:03d}.txt")
        if os.path.isfile(last):
            num_views = n
            break
    else:
        raise FileNotFoundError(
            f"Expected pos_049.txt or pos_064.txt in {cal_dir}. Check DTU SampleSet calibration."
        )
    for i in range(1, num_views + 1):
        fname = os.path.join(cal_dir, f"pos_{i:03d}.txt")
        projection = np.loadtxt(fname, dtype=np.float32)
        camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[:3]
        camera_mat = camera_mat / camera_mat[2, 2]
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot_mat.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]
        pose = pose[:3]
        camtoworlds.append(pose)
    return camtoworlds

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def cull_mesh(cameras, mesh):
    
    vertices = mesh.vertices
    
    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()
    
    sampled_masks = []
    
    for camera in cameras:
        c2w = (camera.world_view_transform.T).inverse()
        w2c = torch.inverse(c2w).cuda()
        mask = camera.gt_mask
        intrinsic = torch.eye(4)
        intrinsic[0, 0] = camera.Fx
        intrinsic[1, 1] = camera.Fy
        intrinsic[0, 2] = camera.Cx
        intrinsic[1, 2] = camera.Cy
        intrinsic = intrinsic.cuda()

        W, H = camera.image_width, camera.image_height
        
        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = mask[0, :, :].cpu().numpy().astype(np.float32) / 256.
            # maski = torch.from_numpy(binary_dilation(maski, disk(14))).float()[None, None].cuda()
            maski = torch.from_numpy(binary_dilation(maski, disk(6))).float()[None, None].cuda()
            
            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)

            sampled_masks.append(sampled_mask)
        
    sampled_masks = torch.stack(sampled_masks, -1)

    # filter
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)
    
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    return mesh


def assert_results_eval_seed(results: dict, eval_seed: int | None) -> None:
    """When a seed was requested, results.json must explicitly record the same int.

    Missing key, JSON null, or any other value is an error. Do not default-fill
    from the parent command line (that would hide an old/wrong child script).
    """
    if eval_seed is None:
        return
    if "eval_seed" not in results:
        raise RuntimeError(
            "results.json missing required field 'eval_seed' "
            f"(expected explicit integer {int(eval_seed)})"
        )
    child_seed = results["eval_seed"]
    if child_seed is None:
        raise RuntimeError(
            f"results.json eval_seed is null; expected explicit integer {int(eval_seed)}"
        )
    try:
        child_int = int(child_seed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"results.json eval_seed not an int: {child_seed!r}") from exc
    if child_int != int(eval_seed):
        raise RuntimeError(f"results.json eval_seed mismatch: {child_int} vs {int(eval_seed)}")


def evaluate_mesh(
    dataset: ModelParams,
    iteration: int,
    DTU_PATH: str,
    mesh_path: str | None = None,
    out_dir: str | None = None,
    eval_seed: int | None = None,
    forbid_roots: list[str] | None = None,
):
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    train_cameras = scene.getTrainCameras()
    dtu_cameras = load_dtu_camera(DTU_PATH)

    gt_points = np.array([cam[:, 3] for cam in dtu_cameras])

    points = []
    for cam in train_cameras:
        c2w = (cam.world_view_transform.T).inverse()
        points.append(c2w[:3, 3].cpu().numpy())
    points = np.array(points)
    gt_points = gt_points[: points.shape[0]]

    # align the scale of two point clouds
    scale_points = np.linalg.norm(points - points.mean(axis=0), axis=1).mean()
    scale_gt_points = np.linalg.norm(gt_points - gt_points.mean(axis=0), axis=1).mean()

    points = points * scale_gt_points / scale_points
    _, r, t = best_fit_transform(points, gt_points)

    # load mesh (cameras/model still from dataset.model_path)
    mesh_file = mesh_path if mesh_path else os.path.join(dataset.model_path, "recon_post.ply")
    print("load", mesh_file)
    mesh = trimesh.load(mesh_file)

    # align the mesh
    mesh.vertices = mesh.vertices * scale_gt_points / scale_points
    mesh.vertices = mesh.vertices @ r.T + t

    write_root = out_dir if out_dir else dataset.model_path
    explicit_new_path = bool(out_dir)
    if explicit_new_path:
        roots = list(forbid_roots or [])
        roots.append(dataset.model_path)
        write_root = assert_fresh_output_dir(
            write_root,
            artifacts=DEFAULT_EVAL_ARTIFACTS,
            forbidden_roots=roots,
            label="evaluate_dtu_mesh",
        )
    else:
        os.makedirs(write_root, exist_ok=True)

    aligned_mesh_file = os.path.join(write_root, "recon_aligned.ply")
    if explicit_new_path and os.path.exists(aligned_mesh_file):
        raise RuntimeError(f"evaluate_dtu_mesh: refuse overwrite {aligned_mesh_file}")
    mesh.export(aligned_mesh_file)

    # evaluate
    vis_dir = os.path.join(write_root, "vis")
    if explicit_new_path:
        os.makedirs(vis_dir, exist_ok=True)
        results_probe = os.path.join(vis_dir, "results.json")
        if os.path.exists(results_probe):
            raise RuntimeError(f"evaluate_dtu_mesh: refuse overwrite {results_probe}")
    else:
        os.makedirs(vis_dir, exist_ok=True)
    # Parse scan id from source_path (e.g. .../scan105 or ...\scan105)
    src_name = os.path.basename(os.path.normpath(dataset.source_path))
    if src_name.lower().startswith("scan"):
        scan = int(src_name[4:].lstrip("_"))
    else:
        raise ValueError(
            f"Cannot get DTU scan id from source_path '{dataset.source_path}'. Expected path ending with 'scan105' etc."
        )

    eval_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "dtu_eval", "eval.py"))
    if not os.path.isfile(eval_script):
        raise FileNotFoundError(f"missing eval script: {eval_script}")

    cmd = [
        sys.executable,
        eval_script,
        "--data",
        os.path.abspath(aligned_mesh_file),
        "--scan",
        str(scan),
        "--mode",
        "mesh",
        "--dataset_dir",
        os.path.abspath(DTU_PATH),
        "--vis_out_dir",
        os.path.abspath(vis_dir),
    ]
    if eval_seed is not None:
        cmd.extend(["--seed", str(int(eval_seed))])
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "eval_seed": None if eval_seed is None else int(eval_seed),
        "mesh_file": os.path.abspath(mesh_file),
        "aligned_mesh_file": os.path.abspath(aligned_mesh_file),
        "vis_dir": os.path.abspath(vis_dir),
        "python": sys.executable,
        "eval_script": eval_script,
        "explicit_new_path": explicit_new_path,
    }
    if completed.returncode != 0:
        receipt_path = os.path.join(write_root, "eval_receipt.json")
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        raise RuntimeError(f"dtu_eval/eval.py failed with exit code {completed.returncode}")

    results_json = os.path.join(vis_dir, "results.json")
    if not os.path.isfile(results_json):
        raise RuntimeError(f"missing results.json after eval: {results_json}")
    with open(results_json, encoding="utf-8") as f:
        results = json.load(f)
    for key in ("mean_d2s", "mean_s2d", "overall"):
        if key not in results or not math.isfinite(float(results[key])):
            raise RuntimeError(f"invalid results.json field {key}: {results.get(key)!r}")
    assert_results_eval_seed(results, eval_seed)
    receipt["results"] = results
    receipt["results_json"] = os.path.abspath(results_json)
    receipt_path = os.path.join(write_root, "eval_receipt.json")
    if explicit_new_path and os.path.exists(receipt_path):
        raise RuntimeError(f"evaluate_dtu_mesh: refuse overwrite {receipt_path}")
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"[evaluate_dtu_mesh] wrote {receipt_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30_000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--scan_id", type=str, help="scan id of the input mesh")
    parser.add_argument(
        "--DTU",
        type=str,
        default="dtu_eval/Offical_DTU_Dataset",
        help="path to the GT DTU point clouds",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="",
        help="Optional explicit recon_post.ply path; cameras still load from -m.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Write recon_aligned.ply and vis/ here (default: model_path).",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=None,
        help="Optional seed forwarded to dtu_eval/eval.py --seed. "
        "Omit to preserve historical unseeded child RNG. Rebuild protocol uses 0.",
    )
    parser.add_argument(
        "--forbid_root",
        action="append",
        default=[],
        help="Extra roots that out_dir must not equal/nest (repeatable). model_path always forbidden when --out_dir set.",
    )

    args = get_combined_args(parser)
    print("evaluating " + args.model_path)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

    evaluate_mesh(
        model.extract(args),
        args.iteration,
        args.DTU,
        mesh_path=(args.mesh_path or None),
        out_dir=(args.out_dir or None),
        eval_seed=args.eval_seed,
        forbid_roots=list(args.forbid_root or []),
    )
