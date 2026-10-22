"""
Low-risk geometry-aware soft routing for M1 losses (not scene-ID MoE).

The router sees only local render statistics (depth / alpha pooled gradients and
ambiguity). It outputs soft expert weights per patch; each expert applies a
fixed, interpretable multiplier vector to
  (lambda_occ, lambda_depth_occ, lambda_depth_occ_rank, lambda_occ_thin).

This keeps the Gaussian backbone unchanged and differs from typical MoE that
splits parameters or representation — here we only schedule physically grounded
M1 terms by geometric mode (flat / boundary / thin-cavity proxy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rows: [occ, depth_occ, rank, thin] multipliers per expert (interpretable presets).
_EXPERT_SCALE_RAW = torch.tensor(
    [
        [0.72, 0.72, 0.65, 0.65],  # smooth / flat — weaken all M1
        [1.02, 1.32, 1.22, 1.08],  # strong boundary — boost depth-occ + rank
        [1.08, 1.05, 0.92, 1.58],  # thin / concavity proxy — boost thin term
    ],
    dtype=torch.float32,
)


def _normalize_expert_columns(scales: torch.Tensor) -> torch.Tensor:
    """Force column means to 1 so uniform routing ≈ baseline lambdas."""
    col_mean = scales.mean(dim=0, keepdim=True).clamp(min=1e-6)
    return scales / col_mean


def compute_m1_router_features(
    median_depth: torch.Tensor,
    alpha: torch.Tensor,
    grid_h: int,
    grid_w: int,
    min_alpha: float,
    eps: float = 1e-6,
) -> torch.Tensor | None:
    """
    Build [P, 4] features from pooled maps: mean depth, mean alpha, mean ambiguity,
    mean edge strength (|∇d|+|∇α| on coarse grid).
    Returns None if too little valid signal.
    """
    d = median_depth
    if d.dim() == 3:
        d = d.squeeze(0)
    if alpha.dim() == 3:
        alpha = alpha.squeeze(0)
    a = alpha.clamp(0.0, 1.0)

    valid = (a > min_alpha) & (d > 0)
    if valid.sum() < 64:
        return None

    dv = d[valid]
    d_norm = (d - dv.min()) / (dv.max() - dv.min() + eps)

    amb = 4.0 * a * (1.0 - a)

    # Coarse pools [1,1,H,W]
    d_s = F.adaptive_avg_pool2d(d_norm.unsqueeze(0).unsqueeze(0), (grid_h, grid_w))[0, 0]
    a_s = F.adaptive_avg_pool2d(a.unsqueeze(0).unsqueeze(0), (grid_h, grid_w))[0, 0]
    amb_s = F.adaptive_avg_pool2d(amb.unsqueeze(0).unsqueeze(0), (grid_h, grid_w))[0, 0]

    # Edge on coarse grid
    gx = torch.abs(d_s[:, 1:] - d_s[:, :-1])
    gy = torch.abs(d_s[1:, :] - d_s[:-1, :])
    pad_x = F.pad(gx, (0, 1, 0, 0))
    pad_y = F.pad(gy, (0, 0, 0, 1))
    edge_d = pad_x + pad_y

    gx_a = torch.abs(a_s[:, 1:] - a_s[:, :-1])
    gy_a = torch.abs(a_s[1:, :] - a_s[:-1, :])
    pad_xa = F.pad(gx_a, (0, 1, 0, 0))
    pad_ya = F.pad(gy_a, (0, 0, 0, 1))
    edge_a = pad_xa + pad_ya
    edge = edge_d + edge_a

    feats = torch.stack([d_s, a_s, amb_s, edge], dim=0)  # [4, gh, gw]
    P = grid_h * grid_w
    x = feats.reshape(4, P).t().contiguous()  # [P, 4]
    # Light normalization for stable MLP inputs (per forward, detach stats ok — keep grad through x)
    x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 0.1)
    return x


class M1GeometrySoftRouter(nn.Module):
    """Per-patch logits -> softmax over experts; multipliers averaged over patches."""

    def __init__(self, hidden_dim: int = 32, num_experts: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )
        scales = _normalize_expert_columns(_EXPERT_SCALE_RAW.clone())
        self.register_buffer("expert_scales", scales)

    def forward(self, patch_feats: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        patch_feats: [P, 4]
        Returns:
          probs: [P, E]
          mean_mult: [4] effective multiplier per M1 channel (avg over patches)
        """
        logits = self.net(patch_feats)
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        # [P,E] @ [E,4] -> [P,4]
        mult_per_patch = probs @ self.expert_scales
        mean_mult = mult_per_patch.mean(dim=0)
        return probs, mean_mult


def router_balance_loss(probs: torch.Tensor) -> torch.Tensor:
    """Encourage each expert to be used (~uniform mean probability over patches)."""
    e = probs.shape[-1]
    mean_usage = probs.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / e)
    return ((mean_usage - target) ** 2).mean()
