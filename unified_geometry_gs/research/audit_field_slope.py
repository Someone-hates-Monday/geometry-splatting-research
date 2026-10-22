"""CPU calculus audit of the existing field-occupancy slope surrogate.

Compares the documented CUDA STE transcription with the derivative of the
continuous slope evaluated at the moving median root. No CUDA call or training
quality is tested; the finite-bracket forward also differs from that continuum.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.median_solver import log_fields, solve


def at_median(alpha, peaks, rho):
    depth = solve(alpha, peaks, 1 / rho, tolerance=1e-12)["depth"]
    slope = .5 * log_fields(depth, alpha, peaks, 1 / rho)[2]
    return depth, slope


def check_case(alpha, peaks, widths):
    rho = 1 / widths
    depth, value = at_median(alpha, peaks, rho)
    a, p, r = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (alpha, peaks, rho))
    t = torch.tensor(depth, dtype=torch.float64, requires_grad=True)
    delta = (t - p) * r
    g = a * torch.exp(-.5 * delta.square())
    log_v = .5 * torch.log1p(-g)
    h = -torch.where(t <= p, log_v, torch.log1p(-a) - log_v).sum()
    ht, = torch.autograd.grad(h, t, create_graph=True, retain_graph=True)
    htt, = torch.autograd.grad(ht, t, retain_graph=True)
    h_param = torch.autograd.grad(h, (a, p, r), retain_graph=True)
    ht_param = torch.autograd.grad(ht, (a, p, r), retain_graph=True)
    total = [(.5 * (u - htt * v / ht)).detach().numpy() for u, v in zip(ht_param, h_param)]
    partial = [(.5 * u).detach().numpy() for u in ht_param]

    td = (depth - peaks) * rho
    exp_g = np.exp(-.5 * td**2)
    gt = alpha * exp_g
    one = np.maximum(1 - gt, 1e-6)
    # Exact transcription of the slope-specific local additions at current source.
    d_s_d_gt = .25 / one**2 * np.abs(td) * rho
    d_s_d_td_explicit = .25 * gt / one * np.sign(td) * rho
    ste = [d_s_d_gt * exp_g, -d_s_d_td_explicit * rho,
           .25 * gt / one * np.abs(td) + d_s_d_td_explicit * (depth - peaks)]
    fd = []
    for parameter_id, parameter in enumerate((alpha, peaks, rho)):
        gradient = np.zeros_like(parameter)
        for j in range(len(parameter)):
            step = 1e-5 * max(1., abs(parameter[j]))
            plus = [x.copy() for x in (alpha, peaks, rho)]
            minus = [x.copy() for x in (alpha, peaks, rho)]
            plus[parameter_id][j] += step
            minus[parameter_id][j] -= step
            gradient[j] = (at_median(*plus)[1] - at_median(*minus)[1]) / (2 * step)
        fd.append(gradient)
    errors = [float(np.max(np.abs(x - y))) for x, y in zip(total, fd)]
    assert max(errors) < 2e-5, errors
    shifted_depth, shifted_value = at_median(alpha, peaks + .137, rho)
    assert abs(shifted_depth - depth - .137) < 1e-10
    assert abs(shifted_value - value) < 1e-8
    assert abs(float(total[1].sum())) < 1e-9
    names = ("alpha", "peak", "inverse_width")
    return {"alpha": alpha.tolist(), "peaks": peaks.tolist(), "widths": widths.tolist(),
            "median": depth, "continuous_slope_at_median": value,
            "total_implicit_gradient": {k: v.tolist() for k, v in zip(names, total)},
            "partial_fixed_root_gradient": {k: v.tolist() for k, v in zip(names, partial)},
            "existing_slope_ste_transcription": {k: v.tolist() for k, v in zip(names, ste)},
            "finite_difference_resolved_root": {k: v.tolist() for k, v in zip(names, fd)},
            "total_gradient_max_abs_fd_error": dict(zip(names, errors)),
            "common_peak_translation_gradient_exact": float(total[1].sum()),
            "common_peak_translation_gradient_ste": float(ste[1].sum()),
            "translated_slope_absolute_error": abs(shifted_value - value)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output must be new.")
    cases = [check_case(np.array([a]), np.array([2.]), np.array([.1])) for a in (.6, .8, .9)]
    cases.append(check_case(np.array([.35, .62, .4]), np.array([1.9, 2., 2.12]), np.array([.03, .08, .05])))
    source = ROOT / "submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu"
    result = {"scope": "Continuous 1D synthetic calculus and source STE transcription, not installed CUDA parity or reconstruction quality.",
              "cases": cases, "formula": "S=q H_t(t*,theta); dS/dtheta=q[H_ttheta-H_tt H_theta/H_t], H=-log T, H_t>0 and indicated second derivatives exist (all t* != p_i suffice).",
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), source)},
              "limitations": ["Existing implementation explicitly documents a straight-through estimator.",
                  "Mismatch with the continuous derivative does not establish that STE caused historical quality failures.",
                  "Continuous exact gradient also differs from finite-bracket forward differentiation.",
                  "At H_t=0 the implicit formula is invalid; exact differentiation alone does not resolve the critical point."]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "max_fd_error": max(max(c["total_gradient_max_abs_fd_error"].values()) for c in cases),
                      "translation_gradient_exact_ste": [[c["common_peak_translation_gradient_exact"], c["common_peak_translation_gradient_ste"]] for c in cases]}, indent=2))


if __name__ == "__main__":
    main()
