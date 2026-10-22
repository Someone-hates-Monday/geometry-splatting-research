"""Mesh-oriented occupancy surrogates (route B): tie alpha ambiguity to geometric consistency."""

import torch


def compute_normal_guided_ambiguity_loss(
    alpha: torch.Tensor,
    normal_error_map: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Weighted mean of 4 * alpha * (1 - alpha) on pixels where depth normals are valid.

    Weight is exp(-beta * e) with e = 1 - |n_render · n_depth| (per-pixel normal disagreement).
    Low disagreement -> stronger push toward opaque/transparent alpha (reduces mushy floaters
    where geometry is already self-consistent).
    """
    a = alpha.squeeze()
    err = normal_error_map.squeeze().clamp(min=0.0)
    v = valid_mask.squeeze().bool()

    if not torch.any(v):
        return torch.zeros((), device=a.device, dtype=a.dtype)

    amb = 4.0 * a * (1.0 - a)
    w = torch.exp(-beta * err)
    wf = w.to(dtype=amb.dtype)
    vf = v.to(dtype=amb.dtype)
    num = (wf * amb * vf).sum()
    den = (wf * vf).sum().clamp(min=eps)
    return num / den
