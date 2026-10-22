# MVP Milestones (Unified Geometry + Weighting)

## M0 — Baseline
- Reproduce G-GS: train + mesh extract on 1–2 DTU scenes.
- **Table**: Chamfer ↓, F1 ↑, time.

## M1 — Depth–occupancy unified
- Add L_occ, L_depth_occ from same T(t).
- **Table**: G-GS | +L_occ | +L_depth_occ | full.

## M2 — Confidence + semantic weighting
- Add w_geo, w_sem to geometry terms.
- **Table**: M1 | +w_geo | +w_sem | both.

## M3 — Full pipeline + mesh
- Mesh from unified field; optional post-refine.
- **Table**: config × mesh extraction.
