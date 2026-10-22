"""Compare local-certificate Newton with existing bracket-contraction solvers.

Function counts are not GPU speedups. Both methods include identical initial
quantile brackets; this file keeps all accuracy and unfavorable count cases.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import mpmath as mp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.median_solver import certified_bracket, log_fields, solve
from research.median_newton import solve_newton


def main():
    output = Path(__file__).resolve().parent / "results/median_newton_verification.json"
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(20261012)
    cases = []
    for _ in range(300):
        n = int(rng.integers(1, 100))
        a = rng.uniform(.001, .99, n)
        if (-np.log1p(-a)).sum() <= np.log(2):
            continue
        p = rng.uniform(-10, 10, n)
        w = np.exp(rng.uniform(np.log(.0001), np.log(2), n))
        cases.append((a, p, w, .5, "original_random_family"))
    for alpha in (.5000001, .6, .75, .9, .999):
        cases.append((np.array([alpha]), np.array([0.]), np.array([1.]), .5, "single_edge_case"))
    for q in (.1, .3, .7, .9):
        cases.append((np.array([.5, .5, .8]), np.array([0., 0., 2.]), np.array([.03, .07, .5]), q, "general_quantile"))
    rows = []
    for a, p, w, q, family in cases:
        lo, hi = certified_bracket(a, p, w, q)
        oracle = brentq(lambda t: log_fields(t, a, p, w)[0] - np.log(q), lo, hi, xtol=1e-13)
        reference_kind = "float64_brent"
        if len(a) == 1:
            with mp.workdps(80):
                aa, qq = mp.mpf(float(a[0])), mp.mpf(float(q))
                near = aa >= 1 - qq**2
                gg = 1 - qq**2 if near else 1 - ((1 - aa) / qq)**2
                displacement = mp.mpf(float(w[0])) * mp.sqrt(-2 * mp.log(gg / aa))
                oracle = float(mp.mpf(float(p[0])) + (-displacement if near else displacement))
            reference_kind = "80_digit_single_component_formula"
        new = solve_newton(a, p, w, q)
        old = solve(a, p, w, q)
        bisect = solve(a, p, w, q, newton=False)
        # These are ordinary float64 endpoint checks, not interval arithmetic.
        if not new["bracket"][0] - 1e-12 <= oracle <= new["bracket"][1] + 1e-12:
            failure = {"alpha": a.tolist(), "peaks": p.tolist(), "widths": w.tolist(), "q": q,
                "family": family, "oracle": oracle, "new": new, "old": old,
                "oracle_logT_residual": log_fields(oracle, a, p, w)[0] - np.log(q),
                "endpoint_logT_residuals": [log_fields(t, a, p, w)[0] - np.log(q) for t in new["bracket"]]}
            failure_path = output.with_name("median_newton_failure_v1.json")
            if not failure_path.exists():
                failure_path.write_text(json.dumps(failure, indent=2) + "\n")
            raise AssertionError(f"Bracket mismatch in {family}; see {failure_path.name}")
        assert abs(new["depth"] - oracle) <= new["absolute_error_bound"] + 1e-12
        rows.append({"family": family, "contributors": len(a), "q": q,
            "new_evaluations": new["log_transmittance_evaluations"],
            "old_hybrid_evaluations": old["log_transmittance_evaluations"],
            "bisection_evaluations": bisect["log_transmittance_evaluations"],
            "high_precision_evaluations": new["high_precision_evaluations"], "reference_kind": reference_kind,
            "root_absolute_error": abs(new["depth"] - oracle),
            "local_endpoint_certificate": new["local_endpoint_certificate"]})
    files = [Path(__file__), Path(__file__).with_name("median_newton.py"), Path(__file__).with_name("median_solver.py")]
    original = [r for r in rows if r["family"] == "original_random_family"]
    result = {"seed": 20261012, "all_assertions_passed": True, "cases": len(rows),
        "original_random_cases": len(original),
        "median_original_family_evaluations": {k: float(np.median([r[k] for r in original])) for k in
            ("new_evaluations", "old_hybrid_evaluations", "bisection_evaluations")},
        "new_worse_than_bisection_original_cases": sum(r["new_evaluations"] > r["bisection_evaluations"] for r in original),
        "high_precision_passes_original_family": sum(r["high_precision_evaluations"] for r in original),
        "max_root_absolute_error": max(r["root_absolute_error"] for r in rows), "rows": rows,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "limitations": ["Synthetic float64 CPU checks; counts exclude sorting and unequal derivative costs.",
            "No GPU speedup, depth-mask equivalence, or reconstruction improvement is established.",
            "Ambiguous endpoint signs use 80-digit numerical recomputation, not rigorous interval arithmetic; its cost is separate.",
            "Generic bracketed Newton and endpoint checks are not claimed novel."]}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
