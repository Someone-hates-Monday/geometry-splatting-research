"""
M2: Geometry confidence and semantic weighting.
- w_geo(p): from cycle reprojection error or transmittance shape.
- w_sem(p): from 2D segmentation (sky, transparent, reflection → downweight).
Apply to depth, normal, L_occ, multi-view terms only (not RGB).
"""

from __future__ import annotations

import torch


def compute_w_geo_from_patchmatch(
    *,
    geo_loss: torch.Tensor,
    valid_ratio: torch.Tensor,
    beta: float = 1.0,
    min_w: float = 0.05,
    max_w: float = 1.0,
) -> torch.Tensor:
    """
    Compute a scalar geometry confidence weight from PatchMatch statistics.

    Inputs:
    - geo_loss: scalar (lower is better). Typical scale depends on pixel_noise.
    - valid_ratio: fraction of valid matches in [0,1] (higher is better).

    Mapping:
      w_geo = clamp(valid_ratio * exp(-beta * geo_loss), min_w, max_w)
    """
    g = torch.clamp(geo_loss, min=0.0)
    v = torch.clamp(valid_ratio, 0.0, 1.0)
    w = v * torch.exp(-beta * g)
    return torch.clamp(w, min=min_w, max=max_w)

