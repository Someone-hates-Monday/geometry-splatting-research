"""Runtime-only excluded-camera guards around the frozen training wrappers.

No pixel arithmetic, optimizer, RNG recipe or production source is changed.
The first 42/7 Scene, NCC cameras and filter arguments are checked in actual
execution. No full-training finite-gradient certification is implied.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__, add_help=False)
    p.add_argument('--holdout-mode', choices=('warmup', 'legacy64', 'factor64'), required=True)
    opts, train_args = p.parse_known_args()
    if '--eval' not in train_args:
        raise ValueError('Held-out training requires the actual eval=True loader option.')
    model = Path(train_args[train_args.index('-m') + 1])
    if model.exists():
        raise ValueError('Fresh model directory required.')
    from scene import GaussianModel, Scene
    import gaussian_renderer
    import warp_patch_ncc
    from research.preflight_factor_holdout import validate_pool
    original_init = Scene.__init__
    original_filter = GaussianModel.compute_3D_filter
    original_render = gaussian_renderer.render
    original_ncc = warp_patch_ncc.warp_patch_ncc
    train_names, test_names = set(), set()
    train_camera_ids, test_camera_ids = set(), set()
    record = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'mode': opts.holdout_mode, 'scene_initializations': 0, 'train_render_calls': 0,
              'final_test_report_render_calls': 0, 'filter_calls': 0, 'ncc_calls': 0,
              'ncc_target_names_seen': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limits': ['RGB exclusion during this new training does not undo earlier use of scan55 for research.',
                        'SfM initialization may use all views; held-out RGB tensors are loaded for final reporting.',
                        'Full gradient finiteness is checked only by the separate 20-update smoke.']}
    targets_seen = set()

    def initialize(self, args, *a, **kw):
        if not args.eval:
            raise ValueError('Actual Scene uses eval=False.')
        original_init(self, args, *a, **kw)
        pool = validate_pool(self)
        if record['scene_initializations']:
            raise ValueError('Unexpected second Scene in a training invocation.')
        record['scene_initializations'] += 1
        record['pool'] = pool
        train_names.update(pool['train_names'])
        test_names.update(pool['test_names'])
        train_camera_ids.update(map(id, self.getTrainCameras()))
        test_camera_ids.update(map(id, self.getTestCameras()))

    def require_train(cam):
        if cam is None or id(cam) not in train_camera_ids or cam.image_name not in train_names:
            raise ValueError('A camera outside the audited train pool entered training.')

    def filter3d(self, cameras):
        if [c.image_name for c in cameras] != record['pool']['train_names']:
            raise ValueError('Actual filter invocation has a different camera list.')
        for cam in cameras:
            require_train(cam)
        record['filter_calls'] += 1
        return original_filter(self, cameras)

    def render(cam, *a, **kw):
        if id(cam) in test_camera_ids:
            frame = inspect.currentframe().f_back
            report_frame = None
            for _ in range(10):
                if frame is None:
                    break
                if frame.f_code.co_name == 'training_report' and Path(frame.f_code.co_filename).resolve() == ROOT / 'train.py':
                    report_frame = frame
                    break
                frame = frame.f_back
            if report_frame is None or report_frame.f_locals['iteration'] != 30000 or torch.is_grad_enabled():
                raise ValueError('Held-out RGB render outside the fixed final no-grad report.')
            record['final_test_report_render_calls'] += 1
        else:
            require_train(cam)
            record['train_render_calls'] += 1
        return original_render(cam, *a, **kw)

    def ncc(*a, **kw):
        frame = inspect.currentframe().f_back
        training = None
        for _ in range(16):
            if frame is None:
                break
            if frame.f_code.co_name == 'training' and Path(frame.f_code.co_filename).resolve() == ROOT / 'train.py':
                training = frame
                break
            frame = frame.f_back
        if training is None:
            raise ValueError('NCC call outside the verified training frame.')
        require_train(training.f_locals['viewpoint_cam'])
        neighbor = training.f_locals['nearest_cam']
        require_train(neighbor)
        targets_seen.add(neighbor.image_name)
        record['ncc_calls'] += 1
        return original_ncc(*a, **kw)

    Scene.__init__ = initialize
    GaussianModel.compute_3D_filter = filter3d
    gaussian_renderer.render = render
    warp_patch_ncc.warp_patch_ncc = ncc
    try:
        if opts.holdout_mode == 'warmup':
            from research import train_solver_branch as inner
            inner_args = ['--solver', 'control', *train_args]
        elif opts.holdout_mode == 'legacy64':
            from research import train_corrected_branch as inner
            inner_args = ['--world-variant', 'legacy64', *train_args]
        else:
            from research import train_factor_branch as inner
            inner_args = ['--world-variant', 'factor64', *train_args]
        sys.argv = [str(Path(inner.__file__)), *inner_args]
        inner.main()
        if record['scene_initializations'] != 1 or record['filter_calls'] < 1 or record['train_render_calls'] < 1:
            raise ValueError('Runtime camera guards were not exercised.')
        if opts.holdout_mode != 'warmup' and record['ncc_calls'] < 1:
            raise ValueError('Runtime NCC guard was not exercised.')
        record['status'] = 'complete'
    except BaseException as error:
        record['status'] = 'failed'
        record['error'] = repr(error)
        raise
    finally:
        Scene.__init__ = original_init
        GaussianModel.compute_3D_filter = original_filter
        gaussian_renderer.render = original_render
        warp_patch_ncc.warp_patch_ncc = original_ncc
        record['ncc_target_names_seen'] = sorted(targets_seen)
        record['updated_utc'] = datetime.now(timezone.utc).isoformat()
        model.mkdir(parents=True, exist_ok=True)
        (model / 'holdout_camera_integration.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
