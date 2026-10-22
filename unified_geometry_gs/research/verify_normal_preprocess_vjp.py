"""Independent proof checks for the Gaussian preprocessing normal normalization.

This tests a local derivative discrepancy, not reconstruction quality. It does
not explain geometry-only variability, whose normal upstream gradient is zero.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import sympy as sp
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Fresh output required.')
    fwd=ROOT/'submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu'
    bwd=ROOT/'submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu'
    if 'glm::normalize(cam_normal_vector)' not in fwd.read_text():raise ValueError('Forward anchor changed.')
    anchor='rnorm3df(normal_vector.x, normal_vector.y, normal_vector.z)'
    if bwd.read_text().count(anchor)!=1:raise ValueError('Backward anchor changed.')
    x=sp.symbols('x',real=True);n=sp.Matrix([-x,0,-1])/sp.sqrt(1+x*x)
    g=sp.Matrix([1,0,0]);raw=sp.Matrix([-x,0,-1])
    truth=sp.diff(n[0],x)
    legacy=((sp.eye(3)-n*n.T)*g).dot(raw.diff(x))
    if sp.simplify(legacy-sp.sqrt(1+x*x)*truth)!=0:raise AssertionError('Missing-factor identity failed.')
    value=sp.simplify(truth.subs(x,1));wrong=sp.simplify(legacy.subs(x,1))
    torch.manual_seed(29);rows=[]
    for _ in range(48):
        raw=torch.randn(3,dtype=torch.float64,requires_grad=True)
        raw=raw*torch.exp(torch.randn((),dtype=torch.float64))
        upstream=torch.randn(3,dtype=torch.float64)
        normalized=raw/torch.linalg.vector_norm(raw)
        oracle=torch.autograd.grad((normalized*upstream).sum(),raw)[0]
        norm=torch.linalg.vector_norm(raw)
        tangent=upstream-normalized*(normalized*upstream).sum()
        corrected=tangent/norm
        historical=tangent/torch.linalg.vector_norm(normalized)
        direction=torch.randn(3,dtype=torch.float64);direction/=torch.linalg.vector_norm(direction)
        h=1e-5*float(norm)
        def loss(v):return (v/torch.linalg.vector_norm(v)*upstream).sum()
        fd=float((loss(raw+h*direction)-loss(raw-h*direction))/(2*h))
        analytic=float((corrected*direction).sum())
        denom=max(float(torch.linalg.vector_norm(oracle)),1e-30)
        rows.append({'raw_normal_length':float(norm.detach()),
            'corrected_relative_l2':float(torch.linalg.vector_norm(corrected-oracle).detach())/denom,
            'historical_relative_l2':float(torch.linalg.vector_norm(historical-oracle).detach())/denom,
            'directional_finite_difference_abs_error':abs(fd-analytic)})
    if max(r['corrected_relative_l2'] for r in rows)>1e-12 or max(r['directional_finite_difference_abs_error'] for r in rows)>1e-7:
        raise AssertionError('Independent normalization checks failed.')
    report={'created_utc':datetime.now(timezone.utc).isoformat(),
        'counterexample':{'precision_matrix':[[2,0,1],[0,1,0],[1,0,1]],'camera_position':[0,0,1],
            'parameter':'precision xz=zx=x, evaluated at x=1','loss':'normalized camera normal x component',
            'true_derivative_exact':str(value),'historical_derivative_exact':str(wrong),
            'true_derivative_float':float(value),'historical_derivative_float':float(wrong),
            'historical_to_true_ratio_exact':'sqrt(2)'},
        'missing_factor_identity_exact':True,'rows':rows,
        'max_corrected_relative_l2':max(r['corrected_relative_l2'] for r in rows),
        'max_directional_finite_difference_abs_error':max(r['directional_finite_difference_abs_error'] for r in rows),
        'local_source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (fwd,bwd,Path(__file__))},
        'upstream_source_check':{'checked_date':'2026-10-12','backward_url':'https://raw.githubusercontent.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu',
            'forward_url':'https://raw.githubusercontent.com/HKUST-SAIL/Geometry-Grounded-Gaussian-Splatting/main/submodules/diff-gaussian-rasterization/cuda_rasterizer/render_forward.cu',
            'observation':'Official main source also normalizes cam_normal_vector then computes reciprocal length from normal_vector in backward.',
            'limitation':'Read official raw main pages; not a pinned upstream commit or exhaustive version history.'},
        'limitations':['Correctness of the local Euclidean normalization VJP only; not a complete world-parameter derivative oracle.',
            'Geometry-only loss has zero upstream normal gradient; this cannot explain its observed numerical variability.',
            'No training, quality improvement, originality or missing-factor intent claim.',
            'The positive-definite example has a realizable positive-scale orthogonal eigendecomposition; no singular covariance is required.']}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Exact missing-factor identity, autograd and directional finite differences passed.',flush=True)


if __name__=='__main__':main()
