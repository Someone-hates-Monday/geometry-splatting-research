"""Depth-edge-weighted normal consistency: emphasize corners / sudden depth jumps."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def depth_gradient_magnitude(depth_1hw: torch.Tensor) -> torch.Tensor:
    """Sobel magnitude on median depth. Input [1,H,W], output [H,W]."""
    if depth_1hw.dim() != 3 or depth_1hw.shape[0] != 1:
        raise ValueError(f"expected [1,H,W] depth, got {tuple(depth_1hw.shape)}")
    d = depth_1hw.unsqueeze(0)  # 1,1,H,W
    device, dtype = d.device, d.dtype
    kx = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-2, -1)
    gx = F.conv2d(d, kx, padding=1)
    gy = F.conv2d(d, ky, padding=1)
    mag = torch.sqrt(gx.squeeze() ** 2 + gy.squeeze() ** 2 + 1e-18)
    return mag  # H,W


def edge_weights_from_depth(
    depth_map_1hw: torch.Tensor,
    valid_hw: torch.Tensor,
    boost: float,
    q_low: float,
    q_high: float,
) -> torch.Tensor:
    """
    Weights in [1, 1+boost] on valid pixels; 0 outside valid (numerator zero anyway).

    boost==0 should not call this (caller uses uniform weights).
    """
    mag = depth_gradient_magnitude(depth_map_1hw)
    v = valid_hw.squeeze().bool()
    if boost <= 0:
        return torch.ones_like(mag)

    vals = mag[v]
    if vals.numel() == 0:
        return torch.ones_like(mag)

    lo = torch.quantile(vals.float(), float(q_low))
    hi = torch.quantile(vals.float(), float(q_high))
    span = (hi - lo).clamp(min=1e-8)
    e_hat = ((mag - lo) / span).clamp(0.0, 1.0)
    e_hat = e_hat * v.to(dtype=mag.dtype)
    return 1.0 + float(boost) * e_hat


def depth_normal_loss_weighted(
    normal_error_map_hw: torch.Tensor,
    valid_hw: torch.Tensor,
    weights_hw: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scalar mean error with per-pixel weights (only valid pixels)."""
    v = valid_hw.squeeze().bool().to(dtype=normal_error_map_hw.dtype)
    w = weights_hw.to(dtype=normal_error_map_hw.dtype)
    num = (w * normal_error_map_hw * v).sum()
    den = (w * v).sum().clamp(min=eps)
    return num / den
