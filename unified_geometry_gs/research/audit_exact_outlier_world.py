"""Read-only world-reference audit, with a separately named shared-amplitude model."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import argparse, hashlib, json, zipfile
import numpy as np
import mpmath as mp
from research.exact_outlier_world_reference import ContinuousModel

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    run = ROOT/'research/experiments/exact_outlier_world_v1'
    capture = ROOT/'research/experiments/exact_outlier_capture_v3'
    manifest, summary = read(run/'manifest.json'), read(run/'summary.json')
    assert manifest['status'] == summary['status'] == 'complete'
    counts = dict(sources=0, source_archive=0, artifacts=0, models=0, precision_coefficient_checks=0,
                  root_precision_brackets=0, accepted_opacity_byte_checks=0, shared_amplitude_roots=0)
    with zipfile.ZipFile(run/'sources.zip') as archive:
        for name, h in manifest['source_sha256'].items():
            assert sha(ROOT/name) == h
            assert hashlib.sha256(archive.read(name)).hexdigest() == h
            counts['sources'] += 1
            counts['source_archive'] += 1
    for name, h in manifest['artifacts_sha256'].items():
        assert sha(run/name) == h
        counts['artifacts'] += 1
    for name, h in manifest['model_sha256'].items():
        assert sha(ROOT/name) == h
        counts['models'] += 1
    rows = []
    max_coefficient_check = Decimal(0)
    max_root_check = Decimal(0)
    with localcontext() as context:
        context.prec = 160
        for sr in summary['rows']:
            case = sr['case']
            stored = read(run/f'ray_{case:02d}.json')
            for name, check in sr['precision_checks'].items():
                assert check['root80_status'] == check['root140_status'] == 'finite'
                assert check['root80_bracket_straddles_at140']
                max_root_check = max(max_root_check, abs(Decimal(check['root_depth_difference_80_140'])))
                counts['root_precision_brackets'] += 1
                for error in check.get('max_scaled_coefficient_differences_floor1e_minus60', {}).values():
                    value = Decimal(error)
                    assert value < Decimal('1e-55'), (case, name, value)
                    max_coefficient_check = max(max_coefficient_check, value)
                    counts['precision_coefficient_checks'] += 1
            with np.load(capture/f'ray_{case:02d}.npz', allow_pickle=False) as z:
                mask = z['accepted_mask']
                co = z['conic_opacity'][:,3][mask]
                activated = z['filtered_opacity'].reshape(-1)[mask]
                assert co.dtype == activated.dtype == np.float32
                assert int(np.sum(co.view(np.uint32) != activated.view(np.uint32))) == sr['conic_opacity_vs_filtered_opacity_accepted_bit_different']
                counts['accepted_opacity_byte_checks'] += len(co)
            newroots = {}
            # Geometry is held at the existing saved-activated reference. Only
            # amplitude is changed to the actual common preprocessing output.
            for dps in (80,140):
                with mp.workdps(dps):
                    coeff = stored['references'][str(dps)]['saved_activated']
                    alpha = [min(mp.mpf(float(np.float32(.99))), mp.mpf(float(v))*mp.exp(-mp.mpf(c['energy'])/2))
                             for v, c in zip(co, coeff)]
                    model = ContinuousModel(alpha, [mp.mpf(c['peak']) for c in coeff], [mp.mpf(c['rho']) for c in coeff])
                    newroots[dps] = model.root()
            with mp.workdps(140):
                assert newroots[80]['status'] == newroots[140]['status'] == 'finite'
                low, high = [mp.mpf(t) for t in newroots[80]['bracket']]
                signs = [model.residual(low), model.residual(high)]
                assert signs[0] < 0 < signs[1]
                reference = mp.mpf(newroots[140]['camera_ray_depth'])
                differences = {arm: mp.mpf(stored['roots']['140'][arm]['camera_ray_depth'])-reference
                               for arm in ('packet','factor')}
                closer = min(differences, key=lambda arm: abs(differences[arm]))
                row = {'case': case, 'scan': sr['scan'], 'camera': sr['camera'], 'pixel': sr['pixel'],
                       'saved_activated_closer': sr['comparisons']['saved_activated']['closer_captured_coefficient_model_under_this_reference'],
                       'raw_physical_closer': sr['comparisons']['raw_physical']['closer_captured_coefficient_model_under_this_reference'],
                       'shared_amplitude_reference_definition': 'Saved-activated covariance inverse geometry, actual common F32 conic_opacity.w, unrounded MP ray alpha with F32 .99 cap.',
                       'shared_amplitude_roots': {str(k):v for k,v in newroots.items()},
                       'shared_amplitude_root80_bracket_signs_at140': [mp.nstr(v, 80) for v in signs],
                       'shared_amplitude_closer': closer,
                       'shared_amplitude_camera_ray_root_differences': {k:mp.nstr(v,80) for k,v in differences.items()},
                       'shared_amplitude_minus_filtered_amplitude_reference_root': mp.nstr(reference-mp.mpf(stored['roots']['140']['saved_activated']['camera_ray_depth']),80),
                       'max_abs_common_opacity_minus_activated_opacity': float(np.max(np.abs(co.astype(float)-activated.astype(float))))}
                counts['shared_amplitude_roots'] += 1
                rows.append(row)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status':'complete', 'CPU_only':True,
              'counts':counts, 'rows':rows,
              'max_MP80_vs_MP140_scaled_coefficient_difference': str(max_coefficient_check),
              'max_MP80_vs_MP140_camera_ray_root_difference': str(max_root_check),
              'closer_counts': {name: {arm:sum(r[name] == arm for r in rows) for arm in ('packet','factor')}
                                for name in ('saved_activated_closer','raw_physical_closer','shared_amplitude_closer')},
              'source_sha256': {str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                                'research/exact_outlier_world_reference.py':sha(ROOT/'research/exact_outlier_world_reference.py'),
                                'research/experiments/exact_outlier_world_v1/manifest.json':sha(run/'manifest.json'),
                                'research/experiments/exact_outlier_world_v1/summary.json':sha(run/'summary.json')},
              'limits':['All nine are selected anomalous rays; closer counts are descriptive, not significance, generic ranking or GT quality.',
                        'Audit recomputes shared-amplitude roots, but imports the general world-root class; not a fully independent root solver.',
                        'World reference uses exact inverse of saved F32 camera affine map; not a new camera calibration.',
                        'MP precision agreement and bracket signs are numerical, not directed interval certificates.']}
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_sha256','limits')},indent=2))


if __name__ == '__main__':
    main()
