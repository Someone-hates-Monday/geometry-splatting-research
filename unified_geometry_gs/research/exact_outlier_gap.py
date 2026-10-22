"""All-nine posthoc gap bounds and scientific curves; CPU only."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import argparse, hashlib, json, math
import numpy as np
import mpmath as mp
from research.exact_outlier_root_oracle import Coefficients

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def txt(x):
    return mp.nstr(x, 50)


def bounds(coeff, lo, hi, mid):
    interior = sum(lo < p < hi for p in coeff.peak)
    beta = Fraction(1)
    left_tail, right_tail = mp.mpf(0), mp.mpf(0)
    for a, p, r, f in zip(coeff.alpha, coeff.peak, coeff.rho, coeff.fractions):
        if p <= lo:
            beta *= 1-f
            left_tail += -mp.log1p(-a*mp.exp(-((lo-p)*r)**2/2))/2
        elif p >= hi:
            right_tail += -mp.log1p(-a*mp.exp(-((hi-p)*r)**2/2))/2
    delta = (Fraction(1, 2)-beta)/beta
    plateau = mp.log1p(mp.mpf(delta.numerator)/delta.denominator)
    lower, upper = plateau-left_tail, plateau+right_tail
    classification = 'peak_inside_interval' if interior else (
        'root_left_of_gap' if lower > 0 else 'root_right_of_gap' if upper < 0 else 'bound_inconclusive')
    return {'interior_peaks': interior, 'prefix_beta_minus_half': txt(mp.mpf((beta-Fraction(1,2)).numerator)/(beta-Fraction(1,2)).denominator),
            'plateau_H_minus_log2': txt(plateau), 'left_tail_bound_at_lo': txt(left_tail),
            'right_tail_bound_at_hi': txt(right_tail), 'uniform_residual_lower_bound': txt(lower),
            'uniform_residual_upper_bound': txt(upper), 'sufficient_classification': classification,
            'midpoint_residual': txt(coeff.residual(mid)), 'midpoint_H_slope': txt(coeff.slopeH(mid))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    capture = ROOT/'research/experiments/exact_outlier_capture_v3'
    oracle_path = ROOT/'research/results/exact_outlier_root_v1.json'
    oracle = read(oracle_path)
    rows = []
    models = {}
    for i, prior in enumerate(oracle['rows']):
        path = capture/f'ray_{i:02d}.npz'
        assert sha(path) == prior['input_sha256']
        with np.load(path, allow_pickle=False) as data:
            models[i] = {arm: [data[f'{arm}_{name}'].copy() for name in ('alpha', 'peak', 'rho')]
                         for arm in ('packet', 'factor')}
        mp.mp.dps = 120
        roots = [mp.mpf(a['minus_infinity_root']['depth']) for a in prior['arms']]
        low, high = min(roots), max(roots)
        width = high-low
        lo, hi, mid = low+width/5, high-width/5, (low+high)/2
        record = {'case': i, 'ray_root_interval': [txt(low), txt(high)], 'central_60_percent_gap': [txt(lo), txt(hi)],
                  'midpoint': txt(mid), 'arms': {}}
        for arm, arrays in models[i].items():
            c = Coefficients(*arrays)
            result = bounds(c, lo, hi, mid)
            with mp.workdps(180):
                # Reconstruct all exact binary inputs at higher precision.
                check = bounds(Coefficients(*arrays), mp.mpf(txt(lo)), mp.mpf(txt(hi)), mp.mpf(txt(mid)))
            result['higher_precision_classification_matches'] = result['sufficient_classification'] == check['sufficient_classification']
            result['higher_precision_midpoint_sign_matches'] = mp.sign(mp.mpf(result['midpoint_residual'])) == mp.sign(mp.mpf(check['midpoint_residual']))
            result['higher_precision_check'] = check
            record['arms'][arm] = result
        rows.append(record)
        print(f'gap {i+1}/9 complete', flush=True)
    assert all(a['higher_precision_classification_matches'] and a['higher_precision_midpoint_sign_matches']
               for r in rows for a in r['arms'].values())
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'complete',
              'source_sha256': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                'research/EXACT_OUTLIER_GAP_PROTOCOL.md': sha(ROOT/'research/EXACT_OUTLIER_GAP_PROTOCOL.md'),
                                'research/exact_outlier_root_oracle.py': sha(ROOT/'research/exact_outlier_root_oracle.py')},
              'input_sha256': {'oracle': sha(oracle_path), **{f'ray_{i:02d}.npz': sha(capture/f'ray_{i:02d}.npz') for i in range(9)}},
              'dps': 120, 'check_dps': 180, 'rows': rows,
              'limits': ['Posthoc selected outliers, not prevalence or GT evidence.',
                         'MP signs are numerical evidence, not outward-rounded interval certification.',
                         'Bounds use a peak-free central interval; pointwise slopes are not uniform lower bounds.',
                         'Neither coefficient model is declared raw-world truth.']}
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), layout='constrained')
    observed = [abs(float(r['delta_decomposition_ray']['observed'])) for r in oracle['rows']]
    represented = [abs(float(r['delta_decomposition_ray']['represented_roots'])) for r in oracle['rows']]
    ids = np.arange(9)
    axes[0].plot(observed, ids-.10, 'o', color='#0072B2', label='Observed output gap', markersize=4)
    axes[0].plot(represented, ids+.10, 'x', color='#D55E00', label='Coefficient-model root gap', markersize=4)
    axes[0].set(xscale='log', xlabel='Absolute packet-factor gap\n(camera-ray model units; log scale)',
                ylabel='Selected outlier case ID', yticks=ids, title='A  All nine selected outliers')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7, loc='lower right')
    curves = {}
    selected = sorted(range(9), key=lambda i: represented[i], reverse=True)[:2]
    for ax, case, letter in zip(axes[1:], selected, ('B', 'C')):
        roots = [float(x['minus_infinity_root']['depth']) for x in oracle['rows'][case]['arms']]
        gap = abs(roots[0]-roots[1])
        grid = np.linspace(min(roots)-.10*gap, max(roots)+.10*gap, 801)
        for arm, color, style in (('packet','#0072B2','-'), ('factor','#D55E00','--')):
            coeff = Coefficients(*models[case][arm])
            values = np.array([coeff.residual64(float(t)) for t in grid])
            curves[f'case{case}_{arm}_residual'] = values
            ax.plot(grid, values, color=color, linestyle=style, label=arm)
            index = 0 if arm == 'packet' else 1
            ax.plot(roots[index], 0, marker='o' if arm == 'packet' else 'x', color=color, markersize=5)
        curves[f'case{case}_ray_grid'] = grid
        ax.axhline(0, color='0.5', linewidth=.7)
        ax.set(yscale='symlog', ylim=None, xlabel='Camera-ray parameter (model units)',
               ylabel='H(t) - ln(2) (dimensionless)', title=f'{letter}  Case {case}: root separation {gap:.4g}')
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
    fig.suptitle('Saved FP32 coefficient models: large root gaps across low-slope regions', fontsize=11)
    fig.savefig(args.out/'gap_evidence.png', dpi=300)
    fig.savefig(args.out/'gap_evidence.svg')
    plt.close(fig)
    np.savez_compressed(args.out/'curve_data.npz', **curves, observed_gap=np.array(observed), represented_gap=np.array(represented))
    exports = {'created_utc': datetime.now(timezone.utc).isoformat(), 'figure_scope': 'Provisional scientific diagnostic; publisher/venue requirements unspecified.',
               'selected_curve_cases': selected, 'selection': 'The two largest represented root gaps among all nine, shown with all-nine overview.',
               'transform': 'Absolute gap in panel A; symlog residual panels B/C, linear threshold 1e-5, 801 model samples, no smoothing.',
               'source_summary_sha256': sha(args.out/'summary.json'),
               'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'dpi': 300, 'size_inches': [10.8, 3.4],
               'alt_text': 'Nine output gaps closely match coefficient-model root gaps. The two largest cases show packet and factor residual curves on opposite sides of zero across a nearly flat middle region.',
               'artifacts_sha256': {n: sha(args.out/n) for n in ('gap_evidence.png','gap_evidence.svg','curve_data.npz')}}
    (args.out/'figure_manifest.json').write_text(json.dumps(exports, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':'complete','rays':9,'arms':18,'classes':[[a['sufficient_classification'] for a in r['arms'].values()] for r in rows]},indent=2))


if __name__ == '__main__':
    main()
