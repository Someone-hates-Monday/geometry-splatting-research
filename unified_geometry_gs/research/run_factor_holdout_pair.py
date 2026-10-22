"""Fresh 42/7 warmup, guarded matched branches and automatic fixed evaluation."""
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
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.run_factor_quality_pair import archive_verified, verify_audit, verify_integration


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def dump(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', type=Path, required=True)
    p.add_argument('--old-pilot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    preflight_path = args.preflight.resolve()
    preflight = read(preflight_path)
    prior_path = args.old_pilot.resolve()
    prior = archive_verified(prior_path)
    if preflight['status'] != 'complete' or preflight['checkpoint_loaded'] or not preflight['pool']['disjoint']:
        raise ValueError('Fresh split preflight failed.')
    if prior['status'] != 'complete' or prior['pilot_gate_pass'] or sha(prior_path / 'paired_quality.json') != prior['quality_results_sha256']:
        raise ValueError('Expected the preserved completed negative all-view pilot.')
    for name, digest in preflight['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Preflight source changed: ' + name)
    for audit in ('factor_preprocess_stress_v1', 'factor_preprocess_regular_v1'):
        verify_audit(ROOT / 'research/experiments' / audit)
    sources = prior['source_sha256'].copy()
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Prior audited recipe changed: ' + name)
    new = ('run_factor_holdout_pair.py', 'train_holdout_branch.py', 'preflight_factor_holdout.py',
           'align_holdout_mesh.py', 'FACTOR_HOLDOUT_PROTOCOL.md', 'run_factor_quality_pair.py')
    for name in new:
        sources['research/' + name] = sha(ROOT / 'research' / name)
    for name in preflight['source_sha256']:
        sources[name] = sha(ROOT / name)
    from research.factor_preprocess_backend import payloads as factor_payloads, load_backend as factor_load
    from research.corrected_preprocess_backend import load_backend as legacy_load
    from research.precise_preprocess_backend import promote
    legacy_data = factor_payloads().copy()
    backward_key = 'cuda_rasterizer/render_backward.cu'
    legacy_data[backward_key] = promote((ROOT / 'submodules/diff-gaussian-rasterization' / backward_key).read_text())
    generated = {'factor64': factor_payloads(), 'legacy64': legacy_data}
    for label, data in generated.items():
        for name, text in data.items():
            if name != 'cuda_rasterizer/render_backward.cu' and text != (ROOT / 'submodules/diff-gaussian-rasterization' / name).read_text():
                raise ValueError('Unexpected forward/config change: ' + label + '/' + name)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(out / 'sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name in sources:
            archive.write(ROOT / name, name)
        archive.write(preflight_path, 'preflight.json')
        for label, data in generated.items():
            for name, text in data.items():
                archive.writestr('generated/' + label + '/' + name, text)
    state = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'preflight',
             'parent_pid': os.getpid(), 'scan': 'scan55', 'seed': 0, 'steps': [],
             'source_sha256': sources, 'source_archive_sha256': sha(out / 'sources.zip'),
             'preflight': str(preflight_path), 'preflight_sha256': sha(preflight_path),
             'old_pilot': str(prior_path), 'old_pilot_results_sha256': sha(prior_path / 'paired_quality.json'),
             'old_pilot_geometry_gate_pass': False,
             'warmup_stop': 6999, 'target_iteration': 30000, 'eval': True,
             'train_names': preflight['pool']['train_names'], 'test_names': preflight['pool']['test_names'],
             'order': ['legacy64', 'factor64'],
             'pilot_gate': {'test_psnr_delta_ge_db': .30, 'test_ssim_delta_ge': 0.,
                            'mean_overall_ratio_le': 1.01, 'gaussian_count_ratio_le': 1.10},
             'limits': ['New exploratory RGB question; the prior geometry pilot remains failed.',
                        'scan55 and all49 views were used in earlier research; not an untouched confirmatory dataset.',
                        'SfM initialization may use all views; RGB training exclusion is enforced in this new run.',
                        'One matched pair, no independent training repetitions, significance or speed claim.',
                        'Combined intervention and adaptive capacity, not single-factor or equal-capacity attribution.',
                        'Gram forward is absent from both quality branches.']}

    def write():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        dump(out / 'manifest.json', state)

    def guard():
        for name, digest in sources.items():
            if sha(ROOT / name) != digest:
                raise ValueError('Queued source changed: ' + name)
        if sha(preflight_path) != state['preflight_sha256']:
            raise ValueError('Preflight changed.')
        for name, digest in preflight['input_sha256'].items():
            if sha(Path(name)) != digest:
                raise ValueError('Input data changed: ' + name)
        if 'common_checkpoint_sha256' in state:
            if sha(checkpoint) != state['common_checkpoint_sha256'] or sha(sidecar) != state['sidecar_sha256']:
                raise ValueError('Fresh common state changed.')

    def run(stage, command):
        guard()
        entry = {'stage': stage, 'status': 'starting', 'argv': command,
                 'started_utc': datetime.now(timezone.utc).isoformat()}
        state['steps'].append(entry)
        state['current_stage'] = stage
        write()
        print('[' + stage + '] starting', flush=True)
        start = time.perf_counter()
        with (out / (stage + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            entry.update(status='running', pid=process.pid)
            write()
            rc = process.wait()
        entry.update(status='complete' if rc == 0 else 'failed', returncode=rc,
                     wall_seconds=time.perf_counter() - start, finished_utc=datetime.now(timezone.utc).isoformat())
        write()
        print('[' + stage + '] exit ' + str(rc), flush=True)
        if rc:
            raise RuntimeError(stage + ' failed; all outputs retained, no automatic retry.')

    source = ROOT.parent / 'dtu/DTU/scan55'
    common = out / 'common'
    checkpoint, sidecar = common / 'chkpnt6999.pth', common / 'chkpnt6999.runtime.pth'

    def training(mode, model, diagnostic=False):
        argv = [sys.executable, 'research/train_holdout_branch.py', '--holdout-mode', mode,
                '-s', str(source), '-m', str(model), '-r', '2', '--dataset', 'dtu', '--eval',
                '--use_decoupled_appearance', '3', '--iterations', '30000', '--regularization_from_iter', '7000',
                '--save_iterations', '30000', '--test_iterations', '30000',
                '--checkpoint_iterations', '6999' if mode == 'warmup' else '30000',
                '--lambda_depth_normal', '0', '--lambda_multi_view_ncc', '.6', '--lambda_multi_view_geo', '.02',
                '--multi_view_ncc_views', '1', '--ncc_mask_thresh', '.9', '--quiet']
        if mode == 'warmup':
            argv += ['--warmup-stop', '6999']
        else:
            argv += ['--branch-sidecar', str(sidecar), '--start_checkpoint', str(checkpoint)]
            if diagnostic:
                argv += ['--diagnostic-steps', '20']
        return argv

    def camera_record(model, full=False):
        rec = read(model / 'holdout_camera_integration.json')
        if rec['status'] != 'complete' or rec['pool'] != preflight['pool']:
            raise ValueError('Actual training camera integration differs from preflight.')
        if rec['final_test_report_render_calls'] != (7 if full else 0):
            raise ValueError('Unexpected held-out report count.')
        if set(rec['ncc_target_names_seen']) & set(state['test_names']):
            raise ValueError('Held-out target was used by NCC.')
        state.setdefault('camera_integration_sha256', {})[str(model)] = sha(model / 'holdout_camera_integration.json')

    def integration(model, label):
        filename = 'corrected_training_integration.json' if label == 'legacy64' else 'factor_training_integration.json'
        return read(model / filename)

    write()
    try:
        guard()
        modules = {'legacy64': legacy_load('legacy64'), 'factor64': factor_load()}
        state['backend_sha256'] = {label: sha(Path(module.__file__)) for label, module in modules.items()}
        for label, module in modules.items():
            for name, text in generated[label].items():
                actual = Path(module.__file__).parent / 'src' / name
                if actual.read_text() != text:
                    raise ValueError('Loaded generated source differs from archive: ' + label + '/' + name)
        state['actual_generated_source_verified'] = True
        state['status'] = 'running'
        write()
        run('warmup', training('warmup', common))
        camera_record(common)
        warm = read(common / 'solver_training_integration.json')
        if not warm['completed'] or warm['ncc_calls'] != 0:
            raise ValueError('Fresh warmup was incomplete or already regularized.')
        state.update(initial_state_sha256=warm['initial_state_sha256'], common_checkpoint_sha256=sha(checkpoint),
                     sidecar_sha256=sha(sidecar), fresh_warmup_verified=True)
        write()
        smoke = {}
        for label in state['order']:
            model = out / (label + '_smoke')
            run('smoke_' + label, training(label, model, True))
            camera_record(model)
            smoke[label] = integration(model, label)
            rec = smoke[label]
            if rec['status'] != 'complete' or rec['steps_observed'] != 20 or rec['nonfinite_gradient_groups']:
                raise ValueError('Finite 20-update smoke failed.')
            if rec['initial_state_sha256'] != state['initial_state_sha256'] or rec['backend_sha256'] != state['backend_sha256'][label]:
                raise ValueError('Smoke initial state/backend mismatch.')
        verify_integration(smoke['factor64'], smoke['legacy64'], True, state['backend_sha256']['factor64'])
        state['smoke_pass'] = True
        state['first_camera'] = smoke['legacy64']['first_camera']
        write()
        arms = {}
        for label in state['order']:
            model = out / label
            run('train_' + label, training(label, model))
            camera_record(model, full=True)
            rec = arms[label] = integration(model, label)
            verify_integration(rec, smoke[label], False, state['backend_sha256'][label])
        verify_integration(arms['factor64'], arms['legacy64'], False, state['backend_sha256']['factor64'])
        state['branch_state_verified'] = True
        state['first_forward_losses_identical'] = True
        write()
        meshes, images, artifacts = {}, {}, {}
        for label in state['order']:
            model = out / label
            run('extract_' + label, [sys.executable, 'mesh_extract.py', '-m', str(model), '--iteration', '30000',
                                   '--voxel_size', '.002', '--trunc_voxel_multiplier', '4', '--depth_max', '8', '--num_cluster', '1'])
            run('align_' + label, [sys.executable, 'research/align_holdout_mesh.py', '--model', str(model)])
            meshes[label] = []
            for seed in (0, 1, 2):
                dest = out / f'eval_{label}_seed{seed}'
                run(f'eval_{label}_seed{seed}', [sys.executable, 'research/evaluate_mesh_seeded.py', '--mesh',
                    str(model / 'recon_aligned.ply'), '--scan', '55', '--seed', str(seed), '--out', str(dest)])
                meshes[label].append(read(dest / 'results.json'))
                artifacts[str(dest / 'results.json')] = sha(dest / 'results.json')
            images[label] = {}
            for split in ('train', 'test'):
                dest = out / f'images_{label}_{split}.json'
                run(f'images_{label}_{split}', [sys.executable, 'research/evaluate_images.py', '--model', str(model),
                    '--iteration', '30000', '--split', split, '--out', str(dest)])
                result = read(dest)
                if [x['camera'] for x in result['rows']] != state[split + '_names']:
                    raise ValueError('Evaluation split/camera order mismatch.')
                images[label][split] = result
                artifacts[str(dest)] = sha(dest)
            for file in ('recon_aligned.ply', 'alignment.json', 'point_cloud/iteration_30000/point_cloud.ply', 'holdout_camera_integration.json'):
                artifacts[str(model / file)] = sha(model / file)
        a, b = [read(out / label / 'alignment.json') for label in state['order']]
        for key in ('scale', 'rotation', 'translation', 'camera_names', 'official_zero_based_ids'):
            if a[key] != b[key] or a[key] != preflight['correct_id_alignment'][key]:
                raise ValueError('Alignment transform differs between arms or from preflight.')
        test_delta = {k: images['factor64']['test']['mean'][k] - images['legacy64']['test']['mean'][k]
                      for k in ('psnr_global_rgb_db', 'psnr_repository_channel_mean_db', 'ssim_repository')}
        overall_ratio = sum(x['overall'] for x in meshes['factor64']) / sum(x['overall'] for x in meshes['legacy64'])
        capacity_ratio = images['factor64']['test']['gaussian_count'] / images['legacy64']['test']['gaussian_count']
        gates = {'test_psnr': test_delta['psnr_global_rgb_db'] >= .30, 'test_ssim': test_delta['ssim_repository'] >= 0,
                 'mesh_noninferiority': overall_ratio <= 1.01, 'capacity': capacity_ratio <= 1.10}
        result = {'mesh_metrics': meshes, 'images': images, 'test_mean_delta_factor_minus_legacy': test_delta,
                  'mean_overall_ratio': overall_ratio, 'gaussian_count_ratio': capacity_ratio,
                  'individual_gates': gates, 'pilot_gate_pass': all(gates.values()),
                  'old_pilot_geometry_gate_pass': False, 'artifact_sha256': artifacts, 'limits': state['limits']}
        dump(out / 'paired_quality.json', result)
        state['quality_results_sha256'] = sha(out / 'paired_quality.json')
        state['pilot_gate_pass'] = result['pilot_gate_pass']
        state['status'], state['current_stage'] = 'complete', None
    except BaseException as error:
        state['status'] = 'failed'
        state['error'] = repr(error)
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
