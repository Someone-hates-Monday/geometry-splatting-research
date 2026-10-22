"""Numerical checks for the bracketing theorem and a nonregular median.

Random tests complement proofs in MEDIAN_SOLVER_THEORY.md. Evaluation counts
exclude sorting and are not GPU speedups or reconstruction quality evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.median_solver import certified_bracket, log_fields, solve


def main():
    rng = np.random.default_rng(20261012)
    rows = []
    for _ in range(300):
        n = int(rng.integers(1, 100))
        a = rng.uniform(.001, .99, n)
        if (-np.log1p(-a)).sum() <= np.log(2):
            continue
        p = rng.uniform(-10, 10, n)
        w = np.exp(rng.uniform(np.log(.0001), np.log(2), n))
        lo, hi = certified_bracket(a, p, w)
        oracle = brentq(lambda t: log_fields(t, a, p, w)[0] + np.log(2), lo, hi, xtol=1e-13)
        hybrid = solve(a, p, w)
        bisect = solve(a, p, w, newton=False)
        assert hybrid["bracket"][0] - 1e-12 <= oracle <= hybrid["bracket"][1] + 1e-12
        assert abs(hybrid["depth"] - oracle) <= hybrid["absolute_error_bound"] + 1e-12
        rows.append({"initial_width": hi - lo, "hybrid_evaluations": hybrid["log_transmittance_evaluations"],
                     "bisection_evaluations": bisect["log_transmittance_evaluations"],
                     "depth_absolute_error": abs(hybrid["depth"] - oracle)})
    critical = []
    for epsilon in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        for sign in (-1, 1):
            a = .75 + sign * epsilon
            # Exact single-Gaussian median with peak=0 and width=1.
            ratio = .75 / a if a >= .75 else (1 - 4 * (1 - a)**2) / a
            t = (-1 if a >= .75 else 1) * np.sqrt(-2 * np.log(ratio))
            trans = np.exp(log_fields(t, np.array([a]), np.array([0.]), np.array([1.]))[0])
            assert abs(trans - .5) < 1e-14
            scaled = abs(t) / np.sqrt(epsilon)
            assert abs(scaled - np.sqrt(8 / 3)) < .01
            critical.append({"alpha": a, "median": float(t), "abs_depth_over_sqrt_epsilon": float(scaled),
                             "abs_secant_derivative": float(abs(t) / epsilon)})
    out = {
        "all_assertions_passed": True, "seed": 20261012, "random_mixtures_tested": len(rows),
        "maximum_root_absolute_error": max(x["depth_absolute_error"] for x in rows),
        "median_hybrid_logT_evaluations": float(np.median([x["hybrid_evaluations"] for x in rows])),
        "median_bisection_logT_evaluations": float(np.median([x["bisection_evaluations"] for x in rows])),
        "critical_single_gaussian": critical,
        "scope": "Synthetic float64 CPU reference. Sorting overhead excluded. No GPU acceleration, real-scene frequency or novelty claim.",
    }
    path = Path(__file__).resolve().parent / "results/median_solver_verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "critical_single_gaussian"}, indent=2))


if __name__ == "__main__":
    main()
