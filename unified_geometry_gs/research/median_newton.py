"""Bracketed Newton with a direct local endpoint certificate.

Generic CPU research prototype. The certificate checks endpoint signs instead
of requiring Newton to contract a distant opposite endpoint. Sorting, field
derivatives, GPU divergence and floating-point rounding still cost work.
"""
from __future__ import annotations

import numpy as np
import mpmath as mp

from research.median_solver import certified_bracket, log_fields


def solve_newton(alpha, peaks, widths, q=.5, tolerance=1e-9, initial_depth=None):
    alpha, peaks, widths = (np.asarray(x, dtype=np.float64) for x in (alpha, peaks, widths))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite tolerance required.")
    lo, hi = certified_bracket(alpha, peaks, widths, q)
    initial = [lo, hi]
    h = -np.log(q)
    evaluations = 0
    high_precision_evaluations = 0

    def evaluate(t):
        nonlocal evaluations, high_precision_evaluations
        lt, _, slope = log_fields(t, alpha, peaks, widths)
        evaluations += 1
        residual = -lt - h
        # Flat roots can make ordinary float64 endpoint signs unreliable. This
        # fallback is a high-precision numerical check, not interval arithmetic.
        if abs(residual) <= 64 * np.finfo(float).eps * max(1., h, len(alpha)):
            high_precision_evaluations += 1
            with mp.workdps(80):
                tt = mp.mpf(float(t))
                components = []
                for aa, pp, ww in zip(alpha, peaks, widths):
                    aa, pp, ww = (mp.mpf(float(x)) for x in (aa, pp, ww))
                    gg = aa * mp.exp(-mp.mpf('.5') * ((tt - pp) / ww)**2)
                    lv = mp.mpf('.5') * mp.log1p(-gg)
                    components.append(-lv if tt <= pp else -mp.log1p(-aa) + lv)
                residual = float(mp.fsum(components) + mp.log(mp.mpf(float(q))))
        return residual, slope

    flo, _ = evaluate(lo)
    fhi, _ = evaluate(hi)
    if flo > 0 or fhi < 0:
        raise ArithmeticError("Floating-point bracket endpoint check failed.")
    t = float(initial_depth) if initial_depth is not None and lo < initial_depth < hi else .5 * (lo + hi)
    newton_streak = bisections = 0
    certified_locally = False
    for _ in range(512):
        if hi - lo <= 2 * tolerance:
            break
        f, slope = evaluate(t)
        if f < 0:
            lo, flo = t, f
        else:
            hi, fhi = t, f
        if f == 0 or (slope > 0 and abs(f) <= tolerance * slope):
            probe_lo, probe_hi = max(lo, t - tolerance), min(hi, t + tolerance)
            probe_flo = flo if probe_lo == lo else evaluate(probe_lo)[0]
            probe_fhi = fhi if probe_hi == hi else evaluate(probe_hi)[0]
            if probe_flo <= 0 <= probe_fhi:
                lo, hi = probe_lo, probe_hi
                certified_locally = True
                break
        candidate = t - f / slope if slope > 0 else np.nan
        if lo < candidate < hi and newton_streak < 4:
            t = float(candidate)
            newton_streak += 1
        else:
            t = .5 * (lo + hi)
            newton_streak = 0
            bisections += 1
    else:
        raise ArithmeticError("Bracket did not converge within 512 steps.")
    return {"depth": .5 * (lo + hi), "bracket": [lo, hi], "initial_bracket": initial,
            "absolute_error_bound": .5 * (hi - lo), "log_transmittance_evaluations": evaluations,
            "high_precision_evaluations": high_precision_evaluations,
            "local_endpoint_certificate": certified_locally, "bisection_steps": bisections}
