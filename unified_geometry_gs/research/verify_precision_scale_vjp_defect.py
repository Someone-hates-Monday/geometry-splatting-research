"""Exact conditional log-scale VJP defect and its tight upstream operator norm.

This is the precision-matrix path, holding rotation, upstream and filter fixed.
It is not a bound on the full renderer gradient or evidence of training gains.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    w, x, y, z = q
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))).reshape(3, 3)


def defect(rotation, scales):
    precision = rotation @ torch.diag(scales.pow(-2)) @ rotation.T
    u = rotation / scales[None, :]
    v = precision @ rotation * scales[None, :]
    blocks = torch.stack([torch.outer(v[:, i], v[:, i]) - torch.outer(u[:, i], u[:, i]) for i in range(3)])
    kernel = blocks.reshape(3, 9)
    gram = kernel @ kernel.T
    values, vectors = torch.linalg.eigh(gram)
    lam = values[-1].clamp_min(0)
    operator = 2 * lam.sqrt()
    if float(lam) > 0:
        witness = (vectors[:, -1, None, None] * blocks).sum(0) / lam.sqrt()
        witness = witness / torch.linalg.vector_norm(witness)
        attained = 2 * torch.linalg.vector_norm(kernel @ witness.reshape(-1))
    else:
        attained = torch.tensor(0., dtype=torch.float64)
    return precision, u, v, blocks, float(operator), float(attained)


def main():
    out = ROOT / 'research/results/precision_scale_vjp_defect_v4.json'
    if out.exists():
        raise ValueError('Preserve existing result.')
    # The fixed non-unit quaternion family from the existing roundtrip note.
    e, a = sp.symbols('e a', positive=True)
    r = sp.Matrix([[-e, 1+e, 0], [1+e, -e, 0], [0, 0, -1-2*e]])
    d = sp.diag(1, a, 1)
    precision = r * d.inv()**2 * r.T
    column = r[:, 1]
    direct = -2 * (column.dot(column)) / a**2
    old = -2 * a**2 * (precision * column).dot(precision * column)
    delta, m = 4*e*(1+e), 1+2*e+2*e**2
    expected = m**2 + delta**2*a**2/2 + delta**2*a**4/4
    symbolic_ratio_pass = sp.factor(old/direct - expected) == 0
    leading_limit = sp.simplify(sp.limit((old/direct-1)/(delta**2*a**4), a, sp.oo))
    if not symbolic_ratio_pass or leading_limit != sp.Rational(1, 4):
        raise AssertionError('Sharp log-scale family failed: ratio difference=' + str(sp.factor(old/direct - expected)) + '; limit=' + str(leading_limit))
    # Algebraic identity for each column uses an arbitrary symmetric H.
    ux, uy, uz, wx, wy, wz = sp.symbols('ux uy uz wx wy wz', real=True)
    h0, h1, h2, h3, h4, h5 = sp.symbols('h0:6', real=True)
    u, w = sp.Matrix([ux, uy, uz]), sp.Matrix([wx, wy, wz])
    h = sp.Matrix([[h0, h1, h2], [h1, h3, h4], [h2, h4, h5]])
    column_identity = sp.expand(((u+w).T*h*(u+w)-u.T*h*u)[0] - (2*(u.T*h*w)[0]+(w.T*h*w)[0])) == 0
    if not column_identity:
        raise AssertionError('Defect expansion failed.')
    generator = torch.Generator().manual_seed(12102)
    checks = []
    for _ in range(64):
        r = torch.randn((3, 3), dtype=torch.float64, generator=generator) + torch.eye(3, dtype=torch.float64)
        ell = torch.randn(3, dtype=torch.float64, generator=generator, requires_grad=True)
        raw = torch.randn((3, 3), dtype=torch.float64, generator=generator)
        h = (raw + raw.T) / 2
        s = ell.exp()
        p = r @ torch.diag(s.pow(-2)) @ r.T
        cov = r @ torch.diag(s*s) @ r.T
        actual = torch.autograd.grad((h*p).sum(), ell, retain_graph=True)[0]
        inverse_upstream = (-p @ h @ p).detach()
        legacy = torch.autograd.grad((inverse_upstream*cov).sum(), ell)[0]
        _, u, v, blocks, operator, attained = defect(r, s.detach())
        direct_formula = -2*torch.sum(u*(h@u), dim=0)
        old_formula = -2*torch.sum(v*(h@v), dim=0)
        error_formula = -2*torch.einsum('iab,ab->i', blocks, h)
        err = max(float(torch.linalg.vector_norm(x-y)/max(float(torch.linalg.vector_norm(x)), 1e-12))
                  for x, y in ((actual, direct_formula), (legacy, old_formula), (legacy-actual, error_formula)))
        tightness = abs(operator-attained)/max(operator, 1e-12)
        if err > 1e-12 or tightness > 1e-12:
            raise AssertionError('Independent AD or operator attainer failed.')
        checks.append({'ad_scaled_error': err, 'operator_attainment_relative_error': tightness})
    rows, inputs = [], {}
    for name in ('world_preprocess_stress_v1', 'factor_regular_inputs_v1'):
        path = ROOT / 'research/experiments' / name / 'results.json'
        inputs[name] = sha(path)
        for index, case in enumerate(json.loads(path.read_text())['cases']):
            q = torch.tensor(case['normalized_device_quaternion'], dtype=torch.float64)
            s = torch.tensor(case['filtered_scales'], dtype=torch.float64)
            r = rotation(q)
            _, u, v, blocks, operator, attained = defect(r, s)
            exact = -2 * (u*u).sum(0)
            old = -2 * (v*v).sum(0)
            relative = float(torch.linalg.vector_norm(old-exact)/torch.linalg.vector_norm(exact))
            attainer_error = abs(operator-attained)/max(operator, 1e-30)
            if attainer_error > 1e-10:
                raise AssertionError('Archived-case operator attainer failed.')
            rows.append({'input': name, 'case': index, 'axis_ratio': float(s.max()/s.min()),
                         'unit_Frobenius_upstream_max_absolute_logscale_error': operator,
                         'operator_attainment_relative_error': attainer_error,
                         'trace_precision_upstream_relative_logscale_vector_error': relative,
                         'trace_precision_component_old_over_direct': (old/exact).tolist()})
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
              'sympy_version': sp.__version__, 'exact_column_defect_identity_pass': column_identity,
              'exact_family_ratio_pass': symbolic_ratio_pass, 'family_leading_relative_error_limit': str(leading_limit),
              'independent_ad_cases': len(checks), 'max_ad_scaled_error': max(x['ad_scaled_error'] for x in checks),
              'max_ad_operator_attainment_error': max(x['operator_attainment_relative_error'] for x in checks),
              'input_sha256': inputs, 'saved_case_count': len(rows), 'rows': rows,
              'max_saved_trace_precision_relative_vector_error': max(x['trace_precision_upstream_relative_logscale_vector_error'] for x in rows),
              'limits': ['Exact real-arithmetic result conditional on fixed R, positive filtered scales and identical symmetric precision upstream H.',
                         'Inverse-chain surrogate -P H P is compared to the actual factor graph; numerical shader errors and other formula defects are outside the theorem.',
                         'Worst-case upstream is constructed, not claimed to be attained in training. Absolute operator norm has scale-dependent units.',
                         'The non-unit quaternion family is not a claim that every member is produced by float32 normalization.',
                         'Raw log-scale gradients also include the filter pullback; other renderer paths may add or cancel errors.',
                         'No complete renderer gradient, quality, speed, statistical significance or novelty claim.']}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'limits')}, indent=2))


if __name__ == '__main__':
    main()
