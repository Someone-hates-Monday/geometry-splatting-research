"""
M1: Unified depth-occupancy losses.

This module implements losses from the same rasterizer outputs:
  - `compute_occ_loss(alpha)`: occupancy regularization from rendered alpha
  - `compute_depth_occ_loss(depth, alpha)`: |∇d| vs |∇α| magnitude coupling
  - `compute_occlusion_ranking_loss(depth, alpha)`: structured M1 — margin ranking
    at α-boundaries (higher-opacity side encouraged to be closer in normalized depth)
  - `compute_thin_occ_boundary_loss(depth, alpha)`: thin-structure helper that
    suppresses ambiguous α around strong geometric boundaries

They are designed to be numerically stable and optional (zero-safe).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_hw(x: torch.Tensor) -> torch.Tensor:
    """Convert [H,W]/[1,H,W]/[B,1,H,W] to [H,W] for loss computation."""
    if x.dim() == 2:
        return x
    if x.dim() == 3:
        return x.squeeze(0)
    if x.dim() == 4:
        return x.squeeze(0).squeeze(0)
    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def compute_occ_loss(
    alpha: torch.Tensor,
    entropy_weight: float = 1.0,
    sparsity_weight: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Occupancy regularization on rendered alpha.

    - Entropy term encourages confident occupancy (close to 0/1).
    - Sparsity term encourages compact occupancy (lower average alpha).
    """
    a = _to_hw(alpha).clamp(eps, 1.0 - eps)
    entropy = -(a * torch.log(a) + (1.0 - a) * torch.log(1.0 - a)).mean()
    sparsity = a.mean()
    return entropy_weight * entropy + sparsity_weight * sparsity


def compute_depth_occ_loss(
    median_depth: torch.Tensor,
    alpha: torch.Tensor,
    min_alpha: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Couple depth discontinuities with occupancy boundaries.

    We match depth-gradient magnitude and alpha-gradient magnitude on valid
    foreground support. This is a practical proxy for depth-occupancy
    consistency from the same rendered transmittance field.
    """
    d = _to_hw(median_depth)
    a = _to_hw(alpha).clamp(0.0, 1.0)

    valid = (a > min_alpha) & (d > 0)
    if valid.sum() < 16:
        return torch.zeros((), dtype=d.dtype, device=d.device)

    d_valid = d[valid]
    d_min = d_valid.min()
    d_max = d_valid.max()
    d_norm = (d - d_min) / (d_max - d_min + eps)

    d_dx = torch.abs(d_norm[:, 1:] - d_norm[:, :-1])
    d_dy = torch.abs(d_norm[1:, :] - d_norm[:-1, :])
    a_dx = torch.abs(a[:, 1:] - a[:, :-1])
    a_dy = torch.abs(a[1:, :] - a[:-1, :])

    v_dx = valid[:, 1:] & valid[:, :-1]
    v_dy = valid[1:, :] & valid[:-1, :]
    if v_dx.sum() + v_dy.sum() == 0:
        return torch.zeros((), dtype=d.dtype, device=d.device)

    loss_dx = torch.abs(d_dx - a_dx)[v_dx].mean() if v_dx.any() else torch.zeros((), dtype=d.dtype, device=d.device)
    loss_dy = torch.abs(d_dy - a_dy)[v_dy].mean() if v_dy.any() else torch.zeros((), dtype=d.dtype, device=d.device)
    return 0.5 * (loss_dx + loss_dy)


def compute_occlusion_ranking_loss(
    median_depth: torch.Tensor,
    alpha: torch.Tensor,
    min_alpha: float = 0.05,
    margin: float = 0.02,
    alpha_soft: float = 0.08,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Occlusion-style ordering at occupancy boundaries.

    For adjacent pixels, the side with higher α is treated as more opaque
    (foreground along the ray). We encourage that side to have smaller
    **image-normalized** median depth than the neighbor (closer surface), via
    softplus(margin + sign(Δα) * (d_r - d_l)) on horizontal/vertical pairs.

    `alpha_soft` soft-weights pairs by |Δα| / (|Δα| + alpha_soft) so flat regions
    contribute little; tiny |Δα| is skipped for stability.
    """
    d = _to_hw(median_depth)
    a = _to_hw(alpha).clamp(0.0, 1.0)

    valid = (a > min_alpha) & (d > 0)
    if valid.sum() < 16:
        return torch.zeros((), dtype=d.dtype, device=d.device)

    d_valid = d[valid]
    d_min = d_valid.min()
    d_max = d_valid.max()
    d_norm = (d - d_min) / (d_max - d_min + eps)

    def _edge_term(d_map: torch.Tensor, a_map: torch.Tensor, v_map: torch.Tensor, dim: str) -> torch.Tensor:
        if dim == "x":
            d0, d1 = d_map[:, :-1], d_map[:, 1:]
            a0, a1 = a_map[:, :-1], a_map[:, 1:]
            v0, v1 = v_map[:, :-1], v_map[:, 1:]
        else:
            d0, d1 = d_map[:-1, :], d_map[1:, :]
            a0, a1 = a_map[:-1, :], a_map[1:, :]
            v0, v1 = v_map[:-1, :], v_map[1:, :]

        pair_ok = v0 & v1
        da = a1 - a0
        dd = d1 - d0
        strong = torch.abs(da) > 1e-5
        m = pair_ok & strong
        if not m.any():
            return torch.zeros((), dtype=d.dtype, device=d.device)

        sign = torch.sign(da)
        viol = margin + sign * dd
        w = torch.abs(da) / (torch.abs(da) + alpha_soft + eps)
        per = F.softplus(viol) * w
        return per[m].mean()

    lx = _edge_term(d_norm, a, valid, "x")
    ly = _edge_term(d_norm, a, valid, "y")
    return 0.5 * (lx + ly)


def compute_thin_occ_boundary_loss(
    median_depth: torch.Tensor,
    alpha: torch.Tensor,
    min_alpha: float = 0.05,
    edge_power: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Encourage crisp occupancy at geometric boundaries.

    Thin structures and deep concave regions are often harmed by "thick" ambiguous
    alpha bands. We penalize alpha ambiguity (a*(1-a)) more where boundary evidence
    is strong (depth/alpha gradients), so occupancy transitions become sharper.
    """
    d = _to_hw(median_depth)
    a = _to_hw(alpha).clamp(0.0, 1.0)

    valid = (a > min_alpha) & (d > 0)
    if valid.sum() < 16:
        return torch.zeros((), dtype=d.dtype, device=d.device)

    d_valid = d[valid]
    d_norm = (d - d_valid.min()) / (d_valid.max() - d_valid.min() + eps)

    d_dx = torch.abs(d_norm[:, 1:] - d_norm[:, :-1])
    d_dy = torch.abs(d_norm[1:, :] - d_norm[:-1, :])
    a_dx = torch.abs(a[:, 1:] - a[:, :-1])
    a_dy = torch.abs(a[1:, :] - a[:-1, :])

    v_dx = valid[:, 1:] & valid[:, :-1]
    v_dy = valid[1:, :] & valid[:-1, :]
    if not (v_dx.any() or v_dy.any()):
        return torch.zeros((), dtype=d.dtype, device=d.device)

    amb = 4.0 * a * (1.0 - a)  # [0,1], max at 0.5
    amb_dx = 0.5 * (amb[:, 1:] + amb[:, :-1])
    amb_dy = 0.5 * (amb[1:, :] + amb[:-1, :])

    w_dx = torch.pow(0.5 * (d_dx + a_dx) + eps, edge_power)
    w_dy = torch.pow(0.5 * (d_dy + a_dy) + eps, edge_power)

    l_dx = (w_dx * amb_dx)[v_dx].mean() if v_dx.any() else torch.zeros((), dtype=d.dtype, device=d.device)
    l_dy = (w_dy * amb_dy)[v_dy].mean() if v_dy.any() else torch.zeros((), dtype=d.dtype, device=d.device)
    return 0.5 * (l_dx + l_dy)
