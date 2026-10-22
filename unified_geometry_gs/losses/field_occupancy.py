"""
A: Field-intrinsic occupancy on the GGGS T=1/2 interface.

Replaces screen-space L_occ (alpha entropy) with a penalty on a shallow
transmittance drop at the median-depth bracket:

  s = |T(d_lo) - T(d_hi)| / (d_hi - d_lo)
  L = E_{valid}[ exp(-β * s / median(s)) ]

Forward slope comes from the rasterizer (mdepth channel 1). Backward uses a
straight-through estimator through analytical |∂T/∂t| (see render_backward.cu).
"""

from __future__ import annotations

import torch


def _to_hw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.squeeze(0)
    if x.dim() == 4:
        return x.squeeze(0).squeeze(0)
    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def compute_field_occ_loss(
    interface_slope: torch.Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1e-6,
    clamp_to_median_mult: float = 10.0,
) -> torch.Tensor:
    """
    Field occupancy loss from per-pixel interface slope s.

    Valid pixels are those with s > 0 (renderer found a T=0.5 crossing).
    """
    s = _to_hw(interface_slope).clamp_min(0.0)
    valid = s > 0
    if valid.sum() < 16:
        return s.sum() * 0.0

    s_v = s[valid]
    med = s_v.median().detach().clamp_min(eps)
    s_hat = (s_v / (med + eps)).clamp(0.0, float(clamp_to_median_mult))
    return torch.exp(-float(beta) * s_hat).mean()
