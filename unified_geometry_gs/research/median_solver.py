"""CPU reference for certified median bracketing of continuous GGGS solids.

This is a mathematical prototype, not a replacement for the CUDA rasterizer.
Sorting costs and GPU divergence must be measured before claiming acceleration.
Assumes a finite collection, 0<alpha<1, sigma>0 and origin at minus infinity.
"""
from __future__ import annotations

import numpy as np


def log_fields(t, alpha, peaks, widths):
    delta = (t - peaks) / widths
    g = alpha * np.exp(-.5 * delta**2)
    lv = .5 * np.log1p(-g)
    lt = np.where(t <= peaks, lv, np.log1p(-alpha) - lv).sum()
    # H=-log T is increasing; H'=0 is possible at coincident peaks.
    slope = np.sum(g * np.abs(delta) / (2 * widths * (1 - g)))
    return float(lt), float(lv.sum()), float(slope)


def weighted_quantile(values, weights, level):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, level, side="left")), len(order) - 1)
    return float(values[order[index]])


def certified_bracket(alpha, peaks, widths, q=.5):
    alpha, peaks, widths = (np.asarray(x, dtype=np.float64) for x in (alpha, peaks, widths))
    if not (alpha.ndim == 1 and alpha.size and alpha.shape == peaks.shape == widths.shape):
        raise ValueError("Expected nonempty vectors of equal shape.")
    if not (np.all(np.isfinite([alpha, peaks, widths])) and np.all((alpha > 0) & (alpha < 1))
            and np.all(widths > 0) and 0 < q < 1):
        raise ValueError("Require finite values, 0<alpha,q<1 and positive widths.")
    weights = -np.log1p(-alpha)
    total = weights.sum()
    h = -np.log(q)
    if total <= h:
        raise ValueError("No finite quantile: final transmittance is at least q.")
    f = h / total
    tail = .25 * min(f, 1 - f)
    k = np.sqrt(-2 * np.log(2 * tail))
    lower = weighted_quantile(peaks - k * widths, weights, (h - tail * total) / (1 - tail))
    upper = weighted_quantile(peaks + k * widths, weights, h / (1 - tail))
    # Float64 safety padding; the mathematical certificate is the two quantile bounds.
    pad = 32 * np.finfo(float).eps * max(1., abs(lower), abs(upper))
    return lower - pad, upper + pad


def solve(alpha, peaks, widths, q=.5, tolerance=1e-9, newton=True):
    alpha, peaks, widths = (np.asarray(x, dtype=np.float64) for x in (alpha, peaks, widths))
    lo, hi = certified_bracket(alpha, peaks, widths, q)
    initial = [lo, hi]
    h = -np.log(q)
    evaluations = 0
    for endpoint, sign in ((lo, -1), (hi, 1)):
        residual = -log_fields(endpoint, alpha, peaks, widths)[0] - h
        evaluations += 1
        if sign * residual < -1e-12:
            raise ArithmeticError("Floating point bracket check failed.")
    t = .5 * (lo + hi)
    while hi - lo > 2 * tolerance:
        lt, _, slope = log_fields(t, alpha, peaks, widths)
        evaluations += 1
        residual = -lt - h
        if residual < 0:
            lo = t
        else:
            hi = t
        candidate = t - residual / slope if newton and slope > 0 else np.nan
        # Always contract the bracket, including near a zero interface slope.
        t = candidate if lo + .1 * (hi - lo) < candidate < hi - .1 * (hi - lo) else .5 * (lo + hi)
    return {"depth": .5 * (lo + hi), "bracket": [lo, hi],
            "absolute_error_bound": .5 * (hi - lo), "initial_bracket": initial,
            "log_transmittance_evaluations": evaluations}
