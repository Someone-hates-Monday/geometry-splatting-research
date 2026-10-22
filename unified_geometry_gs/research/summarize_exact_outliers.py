"""Summarize the posthoc all-nine diagnostics without rerunning CUDA or roots."""
from pathlib import Path
from decimal import Decimal as D, getcontext
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

BASE = Path(__file__).resolve().parent
getcontext().prec = 80


def main():
    out = BASE / 'results/exact_outlier_summary_v1.json'
    if out.exists():
        raise FileExistsError(out)
    root_path = BASE / 'results/exact_outlier_root_v1.json'
    cap_path = BASE / 'experiments/exact_outlier_capture_v3/summary.json'
    sol_path = BASE / 'experiments/exact_outlier_solver_v2/summary.json'
    root, cap, sol = [json.loads(p.read_text(encoding='utf-8'))
                      for p in (root_path, cap_path, sol_path)]
    assert all(r['status'] == 'complete' for r in (root, cap, sol))
    assert len(root['rows']) == len(cap['rows']) == len(sol['rows']) == 9
    rows, slopes, stable_errors, finite_errors = [], [], [], []
    for i, (r, c, s) in enumerate(zip(root['rows'], cap['rows'], sol['rows'])):
        npz = cap_path.parent / f'ray_{i:02d}.npz'
        assert hashlib.sha256(npz.read_bytes()).hexdigest() == r['input_sha256']
        with np.load(npz, allow_pickle=False) as data:
            alpha_diff = float(np.max(np.abs(data['packet_alpha'].astype('float64')
                                           - data['factor_alpha'].astype('float64'))))
        decomp = r['delta_decomposition_ray']
        observed = D(decomp['observed'])
        represented = D(decomp['represented_roots'])
        remainder = D(decomp['remaining_numerical_and_conversion_path'])
        assert abs(observed - represented - remainder) < D('1e-38')
        exact_terms = []
        for a in r['arms']:
            assert a['exact_mass']['full_minus_infinity_finite_root']
            assert a['minus_infinity_root']['status'] == 'finite'
            assert a['minus_infinity_root']['higher_precision_endpoint_sign_check']['strict_signs_pass']
            depth = D(a['minus_infinity_root']['depth'])
            stable_errors.append(abs(D(str(a['stable_double_minus_infinity_root']['depth'])) - depth))
            finite_errors.append(abs(D(a['finite_search_replays']['mp']['depth']) - depth))
            slopes.append(D(a['minus_infinity_root']['minus_T_slope']))
            exact_terms.append(a['terms'])
        assert exact_terms == [c['accepted_inputs']] * 2
        changes = {arm: s['results'][arm]['modes'][2]['z_depth']
                   - s['results'][arm]['original_z_depth'] for arm in ('packet', 'factor')}
        rows.append({'case': i, 'scan': c['scan'], 'camera': c['camera'], 'pixel': c['pixel'],
                     'terms': c['accepted_inputs'], 'observed_z_gap_abs': c['saved_absolute_z_depth_difference'],
                     'observed_ray_gap': str(observed), 'represented_ray_root_gap': str(represented),
                     'remaining_ray_path': str(remainder),
                     'abs_remainder_over_abs_observed': float(abs(remainder / observed)),
                     'max_abs_alpha_coefficient_difference': alpha_diff,
                     'fp64_log_solver_z_changes': changes})
    assert sol['fp32_z_bitwise_matches'] == sol['fp32_slope_bitwise_matches'] == 18
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
              'scope': 'Posthoc all nine outliers; saved F32 coefficient models, not raw-world truth.',
              'input_sha256': {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (root_path, cap_path, sol_path)},
              'rows': rows, 'finite_root_models': 18, 'accepted_terms_total': sum(x['terms'] for x in rows),
              'fp32_cuda_depth_and_slope_bitwise_matches': 18,
              'max_abs_remaining_ray_path': str(max(abs(D(x['remaining_ray_path'])) for x in rows)),
              'max_abs_remainder_over_abs_observed': max(x['abs_remainder_over_abs_observed'] for x in rows),
              'true_minus_T_slope_range': [str(min(slopes)), str(max(slopes))],
              'max_stable_double_vs_mp_root_ray_error': str(max(stable_errors)),
              'max_mp_finite_search_vs_mp_root_ray_error': str(max(finite_errors)),
              'limits': ['Numeric MP brackets are not outward-rounded interval certificates.',
                         'New pinned rebuilt binaries replay all affected old outputs; historical hidden state is not recovered.',
                         'Remainder includes finite search, arithmetic, interpolation, probe context and ray/z conversion.',
                         'Nine selected outliers cannot estimate prevalence or establish GT quality or novelty.',
                         'Neither saved F32 representation is declared the raw-world truth.']}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'limits', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
