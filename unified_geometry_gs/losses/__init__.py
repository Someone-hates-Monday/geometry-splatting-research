"""Loss helpers for unified geometry (M1) and weighted geometry (M2)."""

from .unified_occupancy import (
    compute_depth_occ_loss,
    compute_occ_loss,
    compute_occlusion_ranking_loss,
    compute_thin_occ_boundary_loss,
)
from .field_occupancy import compute_field_occ_loss
from .weighted_geometry import compute_w_geo_from_patchmatch
from .mesh_aware_occ import compute_normal_guided_ambiguity_loss
from .edge_depth_normal import depth_normal_loss_weighted, edge_weights_from_depth
