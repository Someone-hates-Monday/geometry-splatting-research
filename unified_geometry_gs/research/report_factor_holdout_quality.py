"""Independently recompute the completed held-out pilot from actual artifacts."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def local(s):
    return Path('C:/' + s[7:]) if sys.platform == 'win32' and s.startswith('/mnt/c/') else Path(s)


def vertex_count(path):
    with path.open('rb') as handle:
        if handle.readline().strip() != b'ply':
            raise ValueError('Expected PLY model.')
        for _ in range(1000):
            line = handle.readline().decode('ascii').strip()
            if line.startswith('element vertex '):
                return int(line.split()[-1])
            if line == 'end_header':
                break
    raise ValueError('PLY vertex count missing.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('Fresh result report required.')
    run = args.run.resolve()
    state, quality = read(run / 'manifest.json'), read(run / 'paired_quality.json')
    if state['status'] != 'complete' or sha(run / 'paired_quality.json') != state['quality_results_sha256']:
        raise ValueError('Expected complete unchanged quality result.')
    fixed_gate = {'test_psnr_delta_ge_db': .3, 'test_ssim_delta_ge': 0.,
                  'mean_overall_ratio_le': 1.01, 'gaussian_count_ratio_le': 1.1}
    if state['pilot_gate'] != fixed_gate or quality['old_pilot_geometry_gate_pass'] is not False:
        raise ValueError('Predefined gate/history changed.')
    expected = ['warmup', 'smoke_legacy64', 'smoke_factor64', 'train_legacy64', 'train_factor64']
    for label in ('legacy64', 'factor64'):
        expected += ['extract_' + label, 'align_' + label, *[f'eval_{label}_seed{i}' for i in (0, 1, 2)],
                     f'images_{label}_train', f'images_{label}_test']
    if [x['stage'] for x in state['steps']] != expected or any(x['status'] != 'complete' or x['returncode'] != 0 for x in state['steps']):
        raise ValueError('Complete stage sequence differs.')
    preflight = read(local(state['preflight']))
    if sha(local(state['preflight'])) != state['preflight_sha256']:
        raise ValueError('Preflight changed.')
    if sha(run / 'sources.zip') != state['source_archive_sha256']:
        raise ValueError('Archive changed.')
    binaries = {}
    for label, pattern in (('legacy64', 'precise_precise2_*/uggs_precise_precise2_*.so'),
                           ('factor64', 'factor64_*/uggs_factor64_*.so')):
        matches = [x for x in (ROOT / 'research/build').glob(pattern) if sha(x) == state['backend_sha256'][label]]
        if len(matches) != 1:
            raise ValueError('Actual backend bytes not uniquely matched.')
        binaries[label] = matches[0]
    generated_count = 0
    with zipfile.ZipFile(run / 'sources.zip') as archive:
        for name, digest in state['source_sha256'].items():
            if sha(ROOT / name) != digest or hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('Actual/archived source changed: ' + name)
        if hashlib.sha256(archive.read('preflight.json')).hexdigest() != state['preflight_sha256']:
            raise ValueError('Archived preflight differs.')
        for name in archive.namelist():
            if name.startswith('generated/'):
                _, label, relative = name.split('/', 2)
                if (binaries[label].parent / 'src' / relative).read_bytes() != archive.read(name):
                    raise ValueError('Actual generated backend changed: ' + name)
                generated_count += 1
    for name, digest in preflight['input_sha256'].items():
        if sha(local(name)) != digest:
            raise ValueError('Dataset/calibration changed: ' + name)
    for name, digest in quality['artifact_sha256'].items():
        if sha(local(name)) != digest:
            raise ValueError('Actual quality artifact changed: ' + name)
    common = run / 'common'
    cp_sha, side_sha = sha(common / 'chkpnt6999.pth'), sha(common / 'chkpnt6999.runtime.pth')
    if cp_sha != state['common_checkpoint_sha256'] or side_sha != state['sidecar_sha256']:
        raise ValueError('Fresh common state changed.')
    warm = read(common / 'solver_training_integration.json')
    if not warm['completed'] or warm['ncc_calls'] != 0 or warm['initial_state_sha256'] != state['initial_state_sha256']:
        raise ValueError('Warmup state differs.')
    warm_argv = state['steps'][0]['argv']
    if '--eval' not in warm_argv or '--start_checkpoint' in warm_argv:
        raise ValueError('Warmup was not a fresh held-out training invocation.')
    arms, images, mesh_means, alignment, camera_counts = {}, {}, {}, {}, {}
    for label in ('legacy64', 'factor64'):
        model = run / label
        filename = 'corrected_training_integration.json' if label == 'legacy64' else 'factor_training_integration.json'
        rec = read(model / filename)
        arms[label] = rec
        if rec['status'] != 'complete' or rec['steps_observed'] != 23000 or rec['backend_sha256'] != state['backend_sha256'][label]:
            raise ValueError('Full training/update/backend differs.')
        if rec['initial_state_sha256'] != state['initial_state_sha256'] or rec['common_checkpoint_sha256'] != cp_sha or rec['sidecar_sha256'] != side_sha:
            raise ValueError('Full common-state resume differs.')
        smoke = read(run / (label + '_smoke') / filename)
        if smoke['status'] != 'complete' or smoke['steps_observed'] != 20 or smoke['nonfinite_gradient_groups']:
            raise ValueError('Finite smoke failed.')
        for key in ('initial_state_sha256', 'first_camera', 'initial_gaussians', 'first_loss_scalars', 'common_checkpoint_sha256', 'sidecar_sha256'):
            if rec[key] != smoke[key]:
                raise ValueError('Full branch differs from fresh smoke: ' + key)
        cameras = read(model / 'holdout_camera_integration.json')
        camera_counts[label] = {k: cameras[k] for k in ('train_render_calls', 'ncc_calls', 'filter_calls', 'final_test_report_render_calls')}
        if cameras['status'] != 'complete' or cameras['pool'] != preflight['pool'] or cameras['final_test_report_render_calls'] != 7:
            raise ValueError('Actual full training camera isolation differs.')
        if set(cameras['ncc_target_names_seen']) & set(state['test_names']):
            raise ValueError('NCC used a held-out target.')
        cfg_expr = ast.parse((model / 'cfg_args').read_text(), mode='eval').body
        if not isinstance(cfg_expr, ast.Call) or not isinstance(cfg_expr.func, ast.Name) or cfg_expr.func.id != 'Namespace':
            raise ValueError('Unexpected saved configuration format.')
        cfg = {kw.arg: ast.literal_eval(kw.value) for kw in cfg_expr.keywords}
        if cfg['eval'] is not True or cfg['resolution'] != 2 or cfg['white_background'] or cfg['kernel_size'] != 0:
            raise ValueError('Saved split/rendering config differs.')
        a = alignment[label] = read(model / 'alignment.json')
        if sha(model / 'recon_post.ply') != a['source_mesh_sha256'] or sha(model / 'recon_aligned.ply') != a['aligned_mesh_sha256']:
            raise ValueError('Mesh alignment bytes differ.')
        for name, digest in a['source_sha256'].items():
            if sha(local(name)) != digest:
                raise ValueError('Alignment source changed.')
        metrics = []
        for seed in (0, 1, 2):
            ev_dir = run / f'eval_{label}_seed{seed}'
            ev, raw = read(ev_dir / 'manifest.json'), read(ev_dir / 'results.json')
            if ev['status'] != 'complete' or ev['returncode'] != 0 or ev['shuffle_seed'] != seed or ev['scan'] != 55:
                raise ValueError('Evaluator seed/status/scan differs.')
            for name, digest in ev['source_sha256'].items():
                if sha(local(name)) != digest:
                    raise ValueError('Actual evaluator, GT or mesh changed: ' + name)
            if raw != quality['mesh_metrics'][label][seed] or abs(raw['overall'] - .5*(raw['mean_d2s'] + raw['mean_s2d'])) > 1e-12:
                raise ValueError('Raw mesh metrics/identity differ.')
            metrics.append(raw)
        mesh_means[label] = {k: statistics.mean(x[k] for x in metrics) for k in ('overall', 'mean_d2s', 'mean_s2d')}
        images[label] = {}
        count = vertex_count(model / 'point_cloud/iteration_30000/point_cloud.ply')
        for split, expected_count in (('train', 42), ('test', 7)):
            image = read(run / f'images_{label}_{split}.json')
            if image != quality['images'][label][split]:
                raise ValueError('Actual image JSON differs from pair.')
            if image['split'] != split or image['camera_count'] != expected_count or image['partial_smoke'] or image['iteration'] != 30000:
                raise ValueError('Image split/count/iteration differs.')
            if image['is_novel_view_evaluation'] != (split == 'test') or [x['camera'] for x in image['rows']] != state[split + '_names']:
                raise ValueError('Image camera IDs or excluded-view flag differ.')
            if image['gaussian_count'] != count or sha(model / 'cfg_args') != image['source_sha256']['cfg_args'] or sha(model / 'point_cloud/iteration_30000/point_cloud.ply') != image['source_sha256']['point_cloud.ply']:
                raise ValueError('Actual image model/count/config changed.')
            if sha(ROOT / 'research/evaluate_images.py') != image['source_sha256']['evaluate_images.py']:
                raise ValueError('Image evaluation helper changed.')
            for key, mean in image['mean'].items():
                if abs(statistics.mean(x[key] for x in image['rows']) - mean) > 1e-12:
                    raise ValueError('Image mean differs from per-camera values.')
            images[label][split] = image
        extract = next(x['argv'] for x in state['steps'] if x['stage'] == 'extract_' + label)
        for key, value in (('--iteration', '30000'), ('--voxel_size', '.002'), ('--trunc_voxel_multiplier', '4'), ('--depth_max', '8'), ('--num_cluster', '1')):
            if extract[extract.index(key) + 1] != value:
                raise ValueError('Extraction recipe changed.')
    for key in ('initial_state_sha256', 'first_camera', 'initial_gaussians', 'first_loss_scalars', 'common_checkpoint_sha256', 'sidecar_sha256'):
        if arms['legacy64'][key] != arms['factor64'][key]:
            raise ValueError('Matched full branches differ: ' + key)
    for key in ('scale', 'rotation', 'translation', 'camera_names', 'official_zero_based_ids'):
        if alignment['legacy64'][key] != alignment['factor64'][key] or alignment['legacy64'][key] != preflight['correct_id_alignment'][key]:
            raise ValueError('ID alignment transform differs.')
    image_deltas, per_view = {}, {}
    for split in ('train', 'test'):
        a, b = images['legacy64'][split], images['factor64'][split]
        if [(x['camera'], x['width'], x['height']) for x in a['rows']] != [(x['camera'], x['width'], x['height']) for x in b['rows']]:
            raise ValueError('Compared image dimensions/camera order differ.')
        for key in ('background', 'resolution_parameter', 'appearance'):
            if a[key] != b[key]:
                raise ValueError('Image rendering convention differs.')
        image_deltas[split] = {k: b['mean'][k] - a['mean'][k] for k in a['mean']}
        per_view[split] = [{'camera': x['camera'], 'psnr_delta_db': y['psnr_global_rgb_db']-x['psnr_global_rgb_db'],
                            'ssim_delta': y['ssim_repository']-x['ssim_repository']} for x, y in zip(a['rows'], b['rows'])]
    ratio = mesh_means['factor64']['overall'] / mesh_means['legacy64']['overall']
    capacity = images['factor64']['test']['gaussian_count'] / images['legacy64']['test']['gaussian_count']
    gates = {'test_psnr': image_deltas['test']['psnr_global_rgb_db'] >= .3,
             'test_ssim': image_deltas['test']['ssim_repository'] >= 0,
             'mesh_noninferiority': ratio <= 1.01, 'capacity': capacity <= 1.1}
    if gates != quality['individual_gates'] or all(gates.values()) != quality['pilot_gate_pass'] or quality['pilot_gate_pass'] != state['pilot_gate_pass']:
        raise ValueError('Recomputed pilot verdict differs.')
    if abs(ratio-quality['mean_overall_ratio']) > 1e-12 or abs(capacity-quality['gaussian_count_ratio']) > 1e-12:
        raise ValueError('Recomputed mesh/capacity ratios differ.')
    for key in image_deltas['test']:
        if abs(image_deltas['test'][key]-quality['test_mean_delta_factor_minus_legacy'][key]) > 1e-12:
            raise ValueError('Recomputed test difference differs.')
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'run': str(run),
              'actual_sources_models_GT_inputs_and_metrics_verified': True, 'source_count': len(state['source_sha256']),
              'generated_files_verified': generated_count, 'source_archive_sha256': state['source_archive_sha256'],
              'paired_quality_sha256': sha(run / 'paired_quality.json'), 'stage_count': len(state['steps']),
              'complete_initial_state_verified': True, 'fresh_eval_true_warmup_verified': True,
              'runtime_camera_counts': camera_counts, 'matched_ID_alignment_verified': True,
              'mesh_mean_mm': mesh_means, 'mean_mesh_delta_factor_minus_legacy_mm':
              {k: mesh_means['factor64'][k] - mesh_means['legacy64'][k] for k in mesh_means['legacy64']},
              'mean_overall_ratio': ratio, 'gaussian_count_ratio': capacity, 'image_mean_deltas': image_deltas,
              'images': {label: {split: {'mean': x['mean'], 'camera_count': x['camera_count'], 'gaussian_count': x['gaussian_count']}
                         for split, x in images[label].items()} for label in images},
              'per_view_deltas': per_view, 'individual_gates': gates, 'pilot_gate_pass': all(gates.values()),
              'old_pilot_geometry_gate_pass': False, 'limits': state['limits']}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('actual_sources_models_GT_inputs_and_metrics_verified', 'stage_count',
                      'source_count', 'image_mean_deltas', 'mean_overall_ratio', 'individual_gates', 'pilot_gate_pass')}, indent=2))


if __name__ == '__main__':
    main()
