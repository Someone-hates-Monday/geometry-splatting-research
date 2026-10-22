"""Independent spatial/quantile integral checks; no production renderer changes."""
from __future__ import annotations
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]


def transmission(t, peaks, alpha, sigma):
    x = (t-peaks)/sigma
    g = alpha*np.exp(-x*x/2)
    v = np.sqrt(1-g)
    return float(np.prod(np.where(t <= peaks, v, (1-alpha)/v)))


def bracket(peaks, sigma):
    return float(min(peaks)-10*max(sigma)), float(max(peaks)+10*max(sigma))


def quantile(q, peaks, alpha, sigma):
    a, b = bracket(peaks, sigma)
    if not transmission(a, peaks, alpha, sigma) > q > transmission(b, peaks, alpha, sigma):
        raise ValueError('No finite common bracket')
    return brentq(lambda t: transmission(t, peaks, alpha, sigma)-q,
                  a, b, xtol=2e-14, rtol=1e-14)


def spatial_mean(peaks, alpha, sigma, qlo, qhi, derivatives=False):
    lo = quantile(qhi, peaks, alpha, sigma)
    hi = quantile(qlo, peaks, alpha, sigma)
    points = sorted(set([lo, hi]+[float(p) for p in peaks if lo < p < hi]))
    errors = []
    def integrate(fun):
        vals = [quad(fun, a, b, epsabs=2e-12, epsrel=2e-12, limit=200)
                for a, b in zip(points[:-1], points[1:])]
        errors.append(sum(v[1] for v in vals)/(qhi-qlo))
        return sum(v[0] for v in vals)/(qhi-qlo)
    mean = lo+integrate(lambda t: transmission(t, peaks, alpha, sigma)-qlo)
    result = {'mean': mean, 'interval_root_bounds': [lo, hi]}
    if derivatives:
        def partials(t):
            x = (t-peaks)/sigma
            g = alpha*np.exp(-x*x/2)
            T = transmission(t, peaks, alpha, sigma)
            pp = T*abs(t-peaks)*g/(2*sigma*sigma*(1-g))
            aa = T*np.where(t <= peaks, -g/(2*alpha*(1-g)),
                             -1/(1-alpha)+g/(2*alpha*(1-g)))
            return pp, aa
        result['position_gradient'] = [integrate(lambda t: partials(t)[0][i]) for i in range(len(peaks))]
        result['opacity_gradient'] = [integrate(lambda t: partials(t)[1][i]) for i in range(len(peaks))]
    result['quad_reported_max_error'] = max(errors)
    return result


def quantile_mean(peaks, alpha, sigma, qlo, qhi):
    # Independent forward graph: integrate re-solved individual quantiles.
    # Include all peak critical levels and plateau levels as integration splits.
    levels = [transmission(float(p), peaks, alpha, sigma) for p in peaks]
    levels += list(np.cumprod(1-alpha))
    points = sorted(set([qlo, qhi]+[float(x) for x in levels if qlo < x < qhi]))
    vals = [quad(lambda q: quantile(q, peaks, alpha, sigma), a, b,
                 epsabs=2e-11, epsrel=2e-11, limit=200)
            for a, b in zip(points[:-1], points[1:])]
    return sum(v[0] for v in vals)/(qhi-qlo), sum(v[1] for v in vals)/(qhi-qlo)


def high_precision_spatial(peaks, alpha, sigma, qlo, qhi, derivatives=True):
    """60-digit independent scalar product-T forward and opacity VJP."""
    mp.mp.dps = 60
    pp, aa, ss = [[mp.mpf(float(x)) for x in values] for values in (peaks, alpha, sigma)]
    ql, qu = mp.mpf(qlo), mp.mpf(qhi)
    def T(t):
        answer = mp.mpf(1)
        for p, a, s in zip(pp, aa, ss):
            v = mp.sqrt(1-a*mp.exp(-((t-p)/s)**2/2))
            answer *= v if t <= p else (1-a)/v
        return answer
    def root(q):
        a, b = map(mp.mpf, bracket(peaks, sigma))
        for _ in range(220):
            mid = (a+b)/2
            if T(mid) > q:
                a = mid
            else:
                b = mid
        return (a+b)/2
    lo, hi = root(qu), root(ql)
    points = [lo]+[p for p in sorted(pp) if lo < p < hi]+[hi]
    mean = lo+mp.quad(lambda t: T(t)-ql, points)/(qu-ql)
    gradients = []
    for i in range(len(pp) if derivatives else 0):
        def partial(t):
            e = mp.exp(-((t-pp[i])/ss[i])**2/2)
            log_partial = -e/(2*(1-aa[i]*e)) if t <= pp[i] else -1/(1-aa[i])+e/(2*(1-aa[i]*e))
            return T(t)*log_partial
        gradients.append(mp.quad(partial, points)/(qu-ql))
    return {'mean': mp.nstr(mean, 40), 'opacity_gradient': [mp.nstr(g, 40) for g in gradients], 'dps': mp.mp.dps}


def thin_prediction(peaks, alpha, qlo, qhi):
    previous, value = 1., 0.
    for p, a in zip(peaks, alpha):
        current = previous*(1-a)
        mass = max(0., min(qhi, previous)-max(qlo, current))
        value += p*mass/(qhi-qlo)
        previous = current
    if not previous < qlo:
        raise ValueError('Band includes terminal transmission mass')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Fresh result path required')
    warnings.simplefilter('error', IntegrationWarning)
    critical = []
    pp = np.array([-1., 0., 1.])
    aa = np.array([1/5, 39/64, 1/5])
    qlo, qhi = .45, .55
    for e in (.5, .35, .25, .2, .15, .1, .075):
        ss = np.full(3, e)
        row = spatial_mean(pp, aa, ss, qlo, qhi, derivatives=True)
        double_fds, fds = [], []
        for i in range(3):
            step = np.zeros(3); step[i] = 1e-6
            plus = spatial_mean(pp, aa+step, ss, qlo, qhi)['mean']
            minus = spatial_mean(pp, aa-step, ss, qlo, qhi)['mean']
            double_fds.append((plus-minus)/(2e-6))
            # The v1 double-precision mean subtraction could not resolve tiny
            # tail components at the original absolute gate (1e-13). Keep
            # those observations and re-solve the same FD at 60 digits.
            hp_plus = high_precision_spatial(pp,aa+step,ss,qlo,qhi,derivatives=False)
            hp_minus = high_precision_spatial(pp,aa-step,ss,qlo,qhi,derivatives=False)
            fds.append(float((mp.mpf(hp_plus['mean'])-mp.mpf(hp_minus['mean']))/mp.mpf('2e-6')))
        fd_error = max(abs(np.array(fds)-row['opacity_gradient'])/np.maximum(abs(np.array(row['opacity_gradient'])), 1e-8))
        a, b = bracket(pp, ss)
        bound = (b-a)/((qhi-qlo)*(1-aa))
        one_sum_error = abs(sum(row['position_gradient'])-1)
        if min(row['position_gradient']) < -1e-10 or one_sum_error > 1e-8 or fd_error > 1e-5 or np.any(abs(np.array(row['opacity_gradient'])) > bound+1e-8):
            raise AssertionError('Critical band check failed: '+json.dumps({'e':e,**row,'fd':fds,'error':fd_error}))
        row.update({'epsilon': e, 'opacity_fd_step': 1e-6, 'opacity_fd': fds,
                    'double_precision_opacity_fd':double_fds,'fd_forward_precision_digits':60,
                    'opacity_fd_max_scaled_error': float(fd_error), 'position_sum_error': one_sum_error,
                    'opacity_absolute_bounds': bound.tolist(), 'thin_depth_prediction': thin_prediction(pp,aa,qlo,qhi)})
        if e == .5:
            direct, error = quantile_mean(pp, aa, ss, qlo, qhi)
            row['independent_quantile_mean'] = direct
            row['quantile_quad_reported_error'] = error
            row['independent_forward_error'] = abs(direct-row['mean'])
            row['high_precision'] = high_precision_spatial(pp,aa,ss,qlo,qhi)
            if row['independent_forward_error'] > 1e-8 or abs(float(row['high_precision']['mean'])-row['mean']) > 1e-10:
                raise AssertionError('Independent critical forward failed')
        critical.append(row)
        print('critical epsilon='+str(e)+' checked', flush=True)
    pp1, aa1, ss1 = np.array([0.]), np.array([.75]), np.array([.2])
    single = spatial_mean(pp1, aa1, ss1, qlo,qhi, derivatives=True)
    single['independent_quantile_mean'], single['quantile_quad_reported_error'] = quantile_mean(pp1, aa1, ss1,qlo,qhi)
    single['high_precision'] = high_precision_spatial(pp1, aa1, ss1,qlo,qhi)
    single['single_median_regular'] = False
    if abs(single['mean']-single['independent_quantile_mean']) > 1e-8 or not np.all(np.isfinite(single['opacity_gradient'])) or abs(single['position_gradient'][0]-1) > 1e-8:
        raise AssertionError('Band containing singular single-layer median failed')
    plateau = []
    for widths in ((1.,1.),(1.,2.)):
        for band in ((.45,.55),(.40,.55)):
            for e in (.5,.2,.075):
                pp2, aa2, ss2 = np.array([-1.,1.]),np.array([.5,.5]),e*np.array(widths)
                row = spatial_mean(pp2,aa2,ss2,*band,derivatives=True)
                expected = thin_prediction(pp2,aa2,*band)
                row.update({'epsilon':e,'width_factors':widths,'band':band,'thin_depth_prediction':expected,'thin_depth_error':abs(row['mean']-expected)})
                plateau.append(row)
    result = {'created_utc':datetime.now(timezone.utc).isoformat(),
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256':hashlib.sha256((ROOT/'research/QUANTILE_BAND_PROTOCOL.md').read_bytes()).hexdigest(),
              'numpy_version':np.__version__,'scipy_version':scipy.__version__,'mpmath_version':mp.__version__,
              'critical_band': [qlo,qhi], 'critical_cases':critical,'single_layer_critical':single,'plateau_cases':plateau,
              'all_declared_checks_pass':True,
              'failed_implementation_archive':'research/results/quantile_band_failure_v1.zip',
              'implementation_repair':'Keep original 1e-5 / floor 1e-8 gate; use 60-digit re-integrated FD to resolve small components. Original double-precision observations retained.',
              'limits':['Ideal finite-ray model; no training, production CUDA or quality claim.',
                        'Interval quantile mean is an existing statistical functional; application novelty unverified.',
                        'Finite quadrature and finite differences do not prove the theorem.',
                        'Band gradients differ from a median gradient and can blend separated surfaces.']}
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'critical_cases':len(critical),'plateau_cases':len(plateau),'all_declared_checks_pass':True,
                      'max_opacity_fd_error':max(x['opacity_fd_max_scaled_error'] for x in critical)},indent=2),flush=True)


if __name__ == '__main__':
    main()
