"""Independent product-form checks of mixture criticality and slope cusps.

Uses SciPy Brent on the product T, rather than the certified log-domain solver.
These synthetic checks accompany analytic proofs; they are not quality results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq


def transmittance(t, a, p, w):
    g = a * np.exp(-.5 * ((t - p) / w)**2)
    vacancy = np.sqrt(1 - g)
    return float(np.prod(np.where(t <= p, vacancy, (1 - a) / vacancy)))


def root_and_slope(a, p, w):
    root = brentq(lambda t: transmittance(t, a, p, w) - .5,
                  float(p.min() - 1), float(p.max() + 1), xtol=1e-14)
    g = a * np.exp(-.5 * ((root - p) / w)**2)
    slope = .5 * float(np.sum(g * np.abs(root - p) / (2 * w**2 * (1 - g))))
    return root, slope


def main():
    out = Path(__file__).resolve().parent / "results/root_regularity_verification.json"
    if out.exists():
        raise FileExistsError(out)
    a, p, w = np.array([.5, .5]), np.array([.7, .7]), np.array([.03, .07])
    k = float(np.sum(a / (4 * w**2 * (1 - a))))
    coefficient = 1 / np.sqrt(2 * (1 - a[0]) * k)
    assert abs(transmittance(.7, a, p, w) - .5) < 1e-15
    critical_rows = []
    for sign in (-1, 1):
        for epsilon in np.logspace(-3, -8, 6):
            aa = a.copy()
            aa[0] += sign * epsilon
            depth, slope = root_and_slope(aa, p, w)
            asymptotic = -sign * coefficient * np.sqrt(epsilon)
            ratio = (depth - .7) / asymptotic
            assert abs(ratio - 1) < .005
            critical_rows.append({"alpha_0_perturbation": float(sign * epsilon),
                "root_displacement": depth - .7, "square_root_prediction": float(asymptotic),
                "displacement_ratio": float(ratio), "root_slope": slope})

    # A regular median root at one Gaussian's peak. The second component gives
    # H_t>0; the first component's |t-p| still makes the root-slope cusp.
    p, w = np.array([0., .4]), np.array([.1, .4])
    g2 = .8 * np.exp(-.5)
    a = np.array([1 - .25 / (1 - g2), .8])
    depth, baseline = root_and_slope(a, p, w)
    assert abs(depth) < 1e-12 and baseline > 0
    cusp = .5 * a[0] / (2 * w[0]**2 * (1 - a[0]))
    cusp_rows = []
    for epsilon in np.logspace(-3, -7, 5):
        one_sided = []
        root_derivatives = []
        for sign in (-1, 1):
            pp = p.copy()
            pp[0] += sign * epsilon
            shifted_root, shifted_slope = root_and_slope(a, pp, w)
            one_sided.append((shifted_slope - baseline) / (sign * epsilon))
            root_derivatives.append((shifted_root - depth) / (sign * epsilon))
        cusp_rows.append({"peak_step": float(epsilon), "slope_derivative_left": one_sided[0],
            "slope_derivative_right": one_sided[1], "root_derivative_left": root_derivatives[0],
            "root_derivative_right": root_derivatives[1]})
    assert abs(cusp_rows[-1]["slope_derivative_left"] + cusp) < 1e-3
    assert abs(cusp_rows[-1]["slope_derivative_right"] - cusp) < 1e-3
    assert max(abs(cusp_rows[-1][key]) for key in ("root_derivative_left", "root_derivative_right")) < 1e-4
    result = {"scope": "Synthetic product-form Brent checks, not GPU or reconstruction evaluation.",
        "critical_mixture": {"alpha": [.5, .5], "peaks": [.7, .7], "widths": [.03, .07],
            "K": k, "square_root_coefficient": float(coefficient), "rows": critical_rows},
        "regular_root_slope_cusp": {"alpha": a.tolist(), "peaks": p.tolist(), "widths": w.tolist(),
            "baseline_depth": depth, "baseline_slope": baseline, "predicted_one_sided_magnitude": float(cusp),
            "rows": cusp_rows},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"critical_cases": len(critical_rows), "max_asymptotic_ratio_error":
        max(abs(r["displacement_ratio"] - 1) for r in critical_rows),
        "cusp_prediction": float(cusp), "cusp_last_step": cusp_rows[-1]}, indent=2))


if __name__ == "__main__":
    main()
