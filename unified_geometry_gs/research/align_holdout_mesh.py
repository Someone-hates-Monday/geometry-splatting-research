"""Align DTU meshes by camera image ID, including a noncontiguous train split.

The historical first-N helper remains frozen. No mesh or RGB score is used to
fit this transform; only the declared training-camera calibration centers.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fit_centers(centers, truth):
    from evaluate_dtu_mesh import best_fit_transform
    centers, truth = np.asarray(centers, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if centers.shape != truth.shape or centers.ndim != 2 or centers.shape[1] != 3 or len(centers) < 3:
        raise ValueError('Expected matching Nx3 camera centers, N >= 3.')
    if not np.isfinite(centers).all() or not np.isfinite(truth).all():
        raise ValueError('Nonfinite calibration centers.')
    extent = np.linalg.norm(centers - centers.mean(0), axis=1).mean()
    target_extent = np.linalg.norm(truth - truth.mean(0), axis=1).mean()
    if min(extent, target_extent) <= 0:
        raise ValueError('Degenerate camera extent.')
    scale = target_extent / extent
    _, rotation, translation = best_fit_transform(centers * scale, truth)
    residual = centers * scale @ rotation.T + translation - truth
    return {'scale': float(scale), 'rotation': rotation.tolist(), 'translation': translation.tolist(),
            'center_rms_mm': float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
            'target_center_extent_mm': float(target_extent)}


def match_and_fit(names, centers, gt_centers):
    if any(not name.isdecimal() for name in names):
        raise ValueError('DTU image names must be zero-based numeric calibration IDs.')
    ids = [int(name) for name in names]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= len(gt_centers) for i in ids):
        raise ValueError('Duplicate or out-of-range camera IDs.')
    if len(ids) != len(centers):
        raise ValueError('Camera ID and center counts differ.')
    fit = fit_centers(centers, np.asarray(gt_centers)[ids])
    return {**fit, 'camera_names': list(names), 'official_zero_based_ids': ids,
            'convention': 'image_name 0000 maps to pos_001.txt; numeric ID i maps to pos_(i+1).txt.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--iteration', type=int, default=30000)
    args = p.parse_args()
    model = args.model.resolve()
    target = model / 'recon_aligned.ply'
    if target.exists() or (model / 'alignment.json').exists():
        p.error('Fresh alignment outputs required.')
    from scene import GaussianModel, Scene
    from evaluate_dtu_mesh import load_dtu_camera
    from research.benchmark_median_cuda import load_config
    import trimesh
    ds, _ = load_config(model)
    if not ds.eval:
        raise ValueError('This helper is restricted to the new held-out protocol.')
    gt = np.asarray([pose[:, 3] for pose in load_dtu_camera(str(ROOT / 'dtu_eval/Offical_DTU_Dataset'))])
    with tempfile.TemporaryDirectory(prefix='align-heldout-', dir=model.parent) as temp:
        scratch = Path(temp)
        (scratch / 'point_cloud').symlink_to(model / 'point_cloud', target_is_directory=True)
        ds.model_path = str(scratch)
        gs = GaussianModel(ds.sh_degree, ds.sg_degree)
        scene = Scene(ds, gs, load_iteration=args.iteration, shuffle=False)
        cams = scene.getTrainCameras()
        centers = [cam.world_view_transform.T.inverse()[:3, 3].cpu().numpy() for cam in cams]
        record = match_and_fit([cam.image_name for cam in cams], centers, gt)
    source = model / 'recon_post.ply'
    mesh = trimesh.load(source, process=True)
    mesh.vertices = mesh.vertices * record['scale'] @ np.asarray(record['rotation']).T + record['translation']
    mesh.export(target)
    record.update(source_mesh_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  aligned_mesh_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  source_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest()
                                 for x in (Path(__file__), ROOT / 'evaluate_dtu_mesh.py')})
    (model / 'alignment.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
