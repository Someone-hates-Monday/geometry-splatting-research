"""Check a conditional surface-equivalence theorem and a separated-solids example.

The general theorem is proved in MATHEMATICAL_AUDIT.md. Numeric checks illustrate
it; finite sampling alone is never treated as proof of a universal statement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def fields(t, alpha, peaks, widths):
    t = np.asarray(t)[..., None]
    g = alpha * np.exp(-.5 * ((t - peaks) / widths)**2)
    vacancy_i = np.sqrt(1 - g)
    ti = np.where(t <= peaks, vacancy_i, (1 - alpha) / vacancy_i)
    transmittance = ti.prod(-1)
    vacancy = vacancy_i.prod(-1)
    gap = np.where(t > peaks, np.log1p(-g) - np.log1p(-alpha), 0).sum(-1)
    return transmittance, vacancy, gap


def verify():
    rng = np.random.default_rng(20261012)
    max_identity_error = 0.
    for _ in range(200):
        n = int(rng.integers(1, 32))
        a = rng.uniform(.01, .95, n)
        p = rng.uniform(-1, 1, n)
        w = rng.uniform(.01, .3, n)
        t = rng.uniform(-2, 2, 100)
        trans, vacancy, gap = fields(t, a, p, w)
        error = np.max(np.abs(np.log(vacancy) - np.log(trans) - gap))
        max_identity_error = max(max_identity_error, float(error))
        assert np.all(trans <= vacancy + 1e-14)
        assert error < 1e-12
        before = float(np.min(p) - 1)
        tb, vb, gb = fields(before, a, p, w)
        assert abs(tb - vb) < 1e-14 and abs(gb) < 1e-14
    a = np.array([.4, .4])
    p = np.array([0., 1.])
    w = np.array([.03, .03])
    root = brentq(lambda t: fields(t, a, p, w)[0] - .5, -2, 3, xtol=1e-14)
    trans, vacancy, gap = fields(root, a, p, w)
    # For every real t, one peak is at distance >= separation/2. This
    # analytic lower bound, rather than a grid minimum, proves nonexistence.
    separation = float(p[1] - p[0])
    lower_bound_v = float(np.sqrt((1 - .4) * (1 - .4 * np.exp(-separation**2 / (8 * .03**2)))))
    assert lower_bound_v > .5
    assert abs(float(trans) - .5) < 1e-12
    reversed_root = brentq(lambda t: fields(t, a, -p[::-1], w)[0] - .5, -3, 2, xtol=1e-14)
    return {
        "identity_random_check_max_abs_error": max_identity_error,
        "identity": "log(V/T) = sum_{t>peak_i} log((1-G_i(t))/(1-alpha_i)) >= 0",
        "separated_solids": {
            "alpha": a.tolist(), "peaks": p.tolist(), "widths": w.tolist(),
            "transmittance_at_positive_infinity": float(np.prod(1 - a)),
            "median_depth_left_to_right": float(root),
            "median_world_coordinate_right_to_left": float(-reversed_root),
            "vacancy_at_median": float(vacancy), "log_vacancy_transmittance_gap": float(gap),
            "analytic_global_vacancy_lower_bound": lower_bound_v,
            "vacancy_half_level_exists": False,
        },
        "scope": "Independent 1D Gaussian stochastic-solid model. Complements the paper's conditional discussion; not a failure claim about all trained scenes.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results/stochastic_surface_verification.json")
    args = parser.parse_args()
    result = verify()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))
