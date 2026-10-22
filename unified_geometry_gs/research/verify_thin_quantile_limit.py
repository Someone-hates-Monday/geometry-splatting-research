"""High-precision checks of strict, plateau and critical thin-ray quantiles.

This is an ideal finite-ray model, not production CUDA or reconstruction quality.
All inputs and cases are retained; root/finite-difference graphs use product T.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]


def transmittance(t, peaks, alpha, sigma):
    value = mp.mpf(1)
    for p, a, s in zip(peaks, alpha, sigma):
        vacancy = mp.sqrt(1-a*mp.exp(-((t-p)/s)**2/2))
        value *= vacancy if t <= p else (1-a)/vacancy
    return value


def root(peaks, alpha, sigma, q):
    lo, hi = min(peaks)-10*max(sigma), max(peaks)+10*max(sigma)
    if not transmittance(lo, peaks, alpha, sigma) > q > transmittance(hi, peaks, alpha, sigma):
        raise ValueError('Quantile not bracketed.')
    for _ in range(520):
        mid = (lo+hi)/2
        value = transmittance(mid, peaks, alpha, sigma)
        if value > q:
            lo = mid
        elif value < q:
            hi = mid
        else:
            return mid
    return (lo+hi)/2


def slopes(t, peaks, alpha, sigma):
    return [abs(t-p)*a*mp.exp(-((t-p)/s)**2/2)/(2*s*s*(1-a*mp.exp(-((t-p)/s)**2/2)))
            for p, a, s in zip(peaks, alpha, sigma)]


def mpstr(x):
    return mp.nstr(x, 40)


def main():
    out = ROOT / 'research/results/thin_quantile_limit_v2.json'
    if out.exists():
        raise ValueError('Fresh result required.')
    mp.mp.dps = 200
    # E=exp(-1/(2 eps^2)); exact positive symbolic tail, without a
    # floating-point simplification of the symmetric root condition.
    E, eps = sp.symbols('E eps', positive=True)
    outer, middle = sp.Rational(1, 5), sp.Rational(39, 64)
    critical_T = (1-outer)/sp.sqrt(1-outer*E)*sp.sqrt(1-middle)*sp.sqrt(1-outer*E)
    critical_slope = outer*E/(eps**2*(1-outer*E))
    opacity_derivative = -1/(2*(1-middle)*critical_slope)
    expected_opacity = -sp.Rational(32, 5)*eps**2*(1/E-sp.Rational(1, 5))
    x = sp.symbols('x', real=True)
    center_vacancy = sp.sqrt(1-middle*sp.exp(-x*x/(2*eps**2)))
    center_before = sp.diff(-sp.log(center_vacancy), x).subs(x, 0)
    center_after = sp.diff(-sp.log((1-middle)/center_vacancy), x).subs(x, 0)
    left = (1-outer)/sp.sqrt(1-outer*sp.exp(-(x+1)**2/(2*eps**2)))
    right = sp.sqrt(1-outer*sp.exp(-(x-1)**2/(2*eps**2)))
    product_slope = -sp.diff(left*center_vacancy*right, x).subs(x, 0)/sp.Rational(1, 2)
    p1, p2, s1, s2 = sp.symbols('p1 p2 s1 s2', real=True)
    weighted = (s1*p2+s2*p1)/(s1+s2)
    symbolic = {'critical_root_exact': sp.simplify(critical_T-sp.Rational(1, 2)) == 0,
                'critical_opacity_formula_exact': sp.simplify(opacity_derivative-expected_opacity) == 0,
                'critical_slope_from_product_T_exact': sp.simplify(product_slope-critical_slope.subs(E, sp.exp(-1/(2*eps**2)))) == 0,
                'plateau_equal_scaled_distances_exact': sp.factor((weighted-p1)/s1+(weighted-p2)/s2) == 0,
                'critical_position_weights_exact': [sp.simplify((critical_slope/2)/critical_slope-sp.Rational(1, 2)) == 0,
                                                    sp.simplify(center_before) == 0 and sp.simplify(center_after) == 0,
                                                    sp.simplify((critical_slope/2)/critical_slope-sp.Rational(1, 2)) == 0]}
    if not all(value if isinstance(value, bool) else all(value) for value in symbolic.values()):
        raise AssertionError('Exact critical identities failed.')
    critical = []
    peaks = list(map(mp.mpf, (-1, 0, 1)))
    alpha = [mp.mpf(1)/5, mp.mpf(39)/64, mp.mpf(1)/5]
    q = mp.mpf(1)/2
    for raw in ('.5', '.35', '.25', '.2', '.15', '.1', '.075'):
        e = mp.mpf(raw)
        sigma = [e]*3
        T = transmittance(mp.mpf(0), peaks, alpha, sigma)
        root_value = root(peaks, alpha, sigma, q)
        T_t = mp.diff(lambda t: transmittance(t, peaks, alpha, sigma), mp.mpf(0))
        partials = []
        for i in range(3):
            partials.append(mp.diff(lambda value: transmittance(mp.mpf(0), peaks[:i]+[value]+peaks[i+1:], alpha, sigma), peaks[i]))
        weights = [-value/T_t for value in partials]
        T_alpha = mp.diff(lambda value: transmittance(mp.mpf(0), peaks, [alpha[0], value, alpha[2]], sigma), alpha[1])
        actual_alpha = -T_alpha/T_t
        tail = mp.exp(-1/(2*e*e))
        formula_alpha = -mp.mpf(32)/5*e*e*(1/tail-mp.mpf(1)/5)
        # At the critical point the linear neighborhood itself shrinks.
        # Tail-squared alpha steps keep re-solved roots inside that neighborhood.
        h = tail*tail*mp.mpf('1e-8')
        plus = root(peaks, [alpha[0], alpha[1]+h, alpha[2]], sigma, q)
        minus = root(peaks, [alpha[0], alpha[1]-h, alpha[2]], sigma, q)
        finite_difference = (plus-minus)/(2*h)
        derivative_error = abs(actual_alpha/formula_alpha-1)
        fd_error = abs(finite_difference/formula_alpha-1)
        weight_error = max(abs(weights[i]-expected) for i, expected in enumerate((mp.mpf('.5'), mp.mpf(0), mp.mpf('.5'))))
        if abs(T-q) > mp.mpf('1e-180') or abs(root_value) > mp.mpf('1e-140') or derivative_error > mp.mpf('1e-100') or fd_error > mp.mpf('1e-6') or weight_error > mp.mpf('1e-100'):
            raise AssertionError('Critical high-precision check failed at eps=' + raw)
        critical.append({'epsilon': raw, 'root': mpstr(root_value), 'position_weights': list(map(mpstr, weights)),
                         'opacity_derivative': mpstr(actual_alpha), 'log10_abs_opacity_derivative': float(mp.log10(abs(actual_alpha))),
                         'alpha_finite_difference_step': mpstr(h), 'resolved_root_fd_relative_error': mpstr(fd_error),
                         'product_T_derivative_relative_error': mpstr(derivative_error)})
    plateau = []
    for widths in ((1, 1), (1, 2)):
        s = list(map(mp.mpf, widths))
        expected = (s[0]*mp.mpf(1)+s[1]*mp.mpf(-1))/sum(s)
        for raw in ('.5', '.2', '.075'):
            sigma = [mp.mpf(raw)*x for x in s]
            actual = root([mp.mpf(-1), mp.mpf(1)], [q, q], sigma, q)
            if abs(actual-expected) > mp.mpf('1e-100'):
                raise AssertionError('Plateau counterexample failed.')
            plateau.append({'epsilon': raw, 'width_factors': widths, 'root': mpstr(actual),
                            'expected_nonlayer_root': mpstr(expected)})
    # Publish every generated input, including the criteria selecting the strict
    # noncritical subdomain of the theorem; do not select by observed performance.
    rng = random.Random(13013)
    regular, rejected = [], []
    for attempt in range(100):
        p = [0.]
        for _ in range(3):
            p.append(p[-1]+rng.uniform(.5, 1.5))
        a = [rng.uniform(.15, .85) for _ in range(4)]
        sf = [rng.uniform(.5, 1.5) for _ in range(4)]
        cumulative = [1.]
        for value in a:
            cumulative.append(cumulative[-1]*(1-value))
        selected = next((i for i in range(4) if cumulative[i+1] < .5 < cumulative[i]), None)
        reason = None
        if selected is None:
            reason = 'no strict finite crossing'
        elif min(cumulative[selected]-.5, .5-cumulative[selected+1]) < .03:
            reason = 'plateau margin below declared .03'
        elif abs(.5-cumulative[selected]*(1-a[selected])**.5) < .03:
            reason = 'peak critical margin below declared .03'
        row = {'attempt': attempt, 'peaks': p, 'alpha': a, 'width_factors': sf, 'selected_zero_based': selected}
        if reason:
            rejected.append({**row, 'domain_exclusion': reason})
            continue
        pp, aa, ss = [list(map(mp.mpf, values)) for values in (p, a, sf)]
        prev = mp.fprod(1-value for value in aa[:selected])
        r, beta = q/prev, 1-aa[selected]
        z = -mp.sqrt(-2*mp.log((1-r*r)/aa[selected])) if r > mp.sqrt(beta) else mp.sqrt(-2*mp.log((1-(beta/r)**2)/aa[selected]))
        rows = []
        for raw in ('.4', '.2', '.1', '.05', '.025'):
            e = mp.mpf(raw)
            sigma = [e*x for x in ss]
            t = root(pp, aa, sigma, q)
            values = slopes(t, pp, aa, sigma)
            weights = [value/sum(values) for value in values]
            rows.append({'epsilon': raw, 'root': mpstr(t), 'leading_depth': mpstr(pp[selected]+e*ss[selected]*z),
                         'scaled_root_error': mpstr(abs((t-pp[selected])/(e*ss[selected])-z)),
                         'position_weights': list(map(mpstr, weights)),
                         'max_onehot_error': mpstr(max(abs(v-(1 if i==selected else 0)) for i, v in enumerate(weights)))})
        if mp.mpf(rows[-1]['scaled_root_error']) > mp.mpf('1e-8') or mp.mpf(rows[-1]['max_onehot_error']) > mp.mpf('1e-6'):
            raise AssertionError('Declared strict noncritical limit check failed: ' + json.dumps({**row, 'checks': rows}))
        regular.append({**row, 'isolated_scaled_root': mpstr(z), 'checks': rows})
        if len(regular) == 20:
            break
    if len(regular) != 20:
        raise AssertionError('Insufficient declared-domain cases.')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': hashlib.sha256((ROOT/'research/THIN_LIMIT_PROTOCOL.md').read_bytes()).hexdigest(),
              'mpmath_dps': mp.mp.dps, 'mpmath_version': mp.__version__, 'sympy_version': sp.__version__,
              'symbolic_checks': symbolic, 'critical_cases': critical, 'plateau_cases': plateau,
              'regular_cases': regular, 'domain_excluded_inputs': rejected,
              'all_declared_checks_pass': True,
              'limits': ['Ideal fixed finite-ray model; no production CUDA, quantization, masks, floors or reconstruction claim.',
                         'Fixed pixel opacity and peak parameters do not model all consequences of thinning a 3D primitive.',
                         'Synthetic/high-precision checks corroborate proofs and are not real-scene quality evidence.',
                         'No equivalence to complete 2DGS/Triangle Splatting algorithms or novelty claim.']}
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'symbolic_checks': symbolic, 'critical_cases': len(critical), 'plateau_cases': len(plateau),
                      'regular_cases': len(regular), 'domain_excluded_inputs': len(rejected), 'all_declared_checks_pass': True}, indent=2))


if __name__ == '__main__':
    main()
