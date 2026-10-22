"""Execute the actual fresh Scene split, neighbor and filter pools before training.

This uses calibration and input metadata, not model-quality observations. The
old all-view checkpoint is never restored. SfM initialization is transductive.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.align_holdout_mesh import match_and_fit, fit_centers
from research.benchmark_median_cuda import load_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pool(scene):
    train, test = scene.getTrainCameras(), scene.getTestCameras()
    tn, vn = [c.image_name for c in train], [c.image_name for c in test]
    all_names = [f'{i:04d}' for i in range(49)]
    expected_test = all_names[::8]
    expected_train = [n for n in all_names if n not in expected_test]
    if tn != expected_train or vn != expected_test or set(tn) & set(vn):
        raise ValueError('Actual Scene split differs from the fixed 42/7 ID split.')
    neighbors = {}
    for cam in train:
        ids = [int(i) for i in cam.nearest_id]
        if any(i < 0 or i >= len(train) for i in ids):
            raise ValueError('Neighbor does not index the train-camera pool.')
        names = [train[i].image_name for i in ids]
        if set(names) & set(vn):
            raise ValueError('A held-out camera entered the NCC neighbor pool.')
        neighbors[cam.image_name] = names
    return {'train_names': tn, 'test_names': vn, 'neighbors_by_train_name': neighbors,
            'train_count': len(tn), 'test_count': len(vn), 'disjoint': True}


def verify_mapping_algebra():
    # An independent, noncontiguous, shuffled known similarity detects first-N
    # matching; no random seed/state is consumed before the actual Scene load.
    ids = [8, 1, 12, 3, 5, 10]
    official = np.asarray([[i, (i * i) % 17, (i * i * i) % 19] for i in range(16)], dtype=float)
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    scale, translation = 2.5, np.asarray([5., -3., 7.])
    centers = (official[ids] - translation) @ rotation / scale
    fit = match_and_fit([f'{i:04d}' for i in ids], centers, official)
    if not np.allclose(fit['scale'], scale, rtol=0, atol=1e-12) or fit['center_rms_mm'] > 1e-12:
        raise ValueError('ID mapping failed the known similarity case.')
    for names in (['0001'] * len(ids), ['invalid'] * len(ids), ['9999'] * len(ids)):
        try:
            match_and_fit(names, centers, official)
        except ValueError:
            pass
        else:
            raise ValueError('Invalid calibration IDs were accepted.')
    return {'known_similarity_rms': fit['center_rms_mm'], 'invalid_ids_rejected': True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Fresh preflight output required.')
    from scene import GaussianModel, Scene
    from evaluate_dtu_mesh import load_dtu_camera, _find_calibration_dir
    from utils.general_utils import safe_state
    ds, _ = load_config(args.config_model)
    ds.eval = True
    source = Path(ds.source_path)
    if source.name != 'scan55':
        raise ValueError('This predeclared pilot uses scan55 only.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    algebra = verify_mapping_algebra()
    safe_state(True)
    with tempfile.TemporaryDirectory(prefix='holdout-preflight-', dir=args.out.parent.resolve()) as temp:
        ds.model_path = temp
        gs = GaussianModel(ds.sh_degree, ds.sg_degree)
        scene = Scene(ds, gs, shuffle=False)
        pool = validate_pool(scene)
        # This is an actual invocation of the production filter with the audited
        # train pool; no held-out camera is passed to this invocation.
        gs.compute_3D_filter(cameras=scene.getTrainCameras().copy())
        if not torch.isfinite(gs.filter_3D).all():
            raise ValueError('Preflight filter produced nonfinite values.')
        train, test = scene.getTrainCameras(), scene.getTestCameras()
        center = lambda cam: cam.world_view_transform.T.inverse()[:3, 3].cpu().numpy()
        centers = np.asarray([center(c) for c in train])
        gt_root = ROOT / 'dtu_eval/Offical_DTU_Dataset'
        gt = np.asarray([pose[:, 3] for pose in load_dtu_camera(str(gt_root))])
        correct = match_and_fit(pool['train_names'], centers, gt)
        wrong = fit_centers(centers, gt[:len(centers)])
        if correct['center_rms_mm'] >= wrong['center_rms_mm']:
            raise ValueError('Numeric image IDs did not resolve the calibration mismatch.')
        if correct['center_rms_mm'] / correct['target_center_extent_mm'] > .02:
            raise ValueError('Matched centers disagree by >2% of calibration extent; mapping requires investigation.')
        camera_rows = [{'name': c.image_name, 'width': c.image_width, 'height': c.image_height,
                        'center': center(c).tolist()} for c in train + test]
        initial_gaussians = len(gs.get_xyz)
    files = [x for x in (source / 'sparse/0').iterdir() if x.is_file()]
    files += [x for x in (source / ds.images).iterdir() if x.is_file()]
    cal_dir = Path(_find_calibration_dir(str(gt_root)))
    files += [cal_dir / f'pos_{i+1:03d}.txt' for i in range(len(gt))]
    sources = ['research/preflight_factor_holdout.py', 'research/align_holdout_mesh.py',
               'scene/__init__.py', 'scene/dataset_readers.py', 'scene/gaussian_model.py',
               'scene/cameras.py', 'utils/camera_utils.py', 'train.py', 'evaluate_dtu_mesh.py',
               'research/benchmark_median_cuda.py', 'arguments/__init__.py']
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
              'scan': 'scan55', 'pool': pool, 'actual_filter_camera_names': pool['train_names'],
              'camera_metadata': camera_rows, 'fresh_sfm_gaussians': initial_gaussians,
              'correct_id_alignment': correct, 'wrong_first_N_alignment': wrong,
              'mapping_algebra': algebra, 'checkpoint_loaded': False,
              'input_sha256': {str(x.resolve()): sha(x) for x in files},
              'source_sha256': {n: sha(ROOT / n) for n in sources},
              'limits': ['Test RGB is loaded by Scene but excluded from train/neighbor/filter pools.',
                         'Existing COLMAP sparse initialization may incorporate all 49 input views: transductive SfM initialization.',
                         'This preflight is not a training run or a model-quality evaluation.',
                         'Actual training invocations require runtime pool guards; this result alone is not an execution trace of training.']}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'complete', 'train_count': 42, 'test_count': 7,
                      'matched_alignment_rms_mm': correct['center_rms_mm'],
                      'wrong_first_N_rms_mm': wrong['center_rms_mm']}, indent=2))


if __name__ == '__main__':
    main()
