# Window/scene diagnostic: refreshed prior-art boundary

2026-10-21. Single-agent targeted primary-source reading, not a systematic review or novelty
certificate. Public search terms and raw discovery/primary returns are recorded in
`literature/window_scene_prior_work_v15.json`. No local unpublished data/formulas uploaded.
Search mirrors, automatic criticism and unrelated DETRIS packages were not used as evidence.

[Elastic Triangle Splatting v1](https://arxiv.org/html/2608.29106v1), §3.1–3.2,6.1,7.1, already
compares single-sided, bilateral and flexible-boundary kernels. It separates unified-distance
shape experiments from original NVS baselines and reports a small-triangle-filter ablation.
Therefore neither boundary gradient starvation nor disabling small-triangle filtering is a
new general contribution here; its claimed NVS gains are literature reports, not our replication.

[2D Triangle Splatting v3](https://arxiv.org/html/2506.18575v3), §3.1, explicitly uses ray-plane
intersection and barycentric opacity/vertex-color interpolation. It discusses tile sorting and
pixel-local resorting. Consequently switching to barycentric kernels, vertex interpolation
or more accurate ordering cannot be presented as a fresh idea from our three-pixel trace.
It is also not source-equivalent to the screen-distance TGS kernel used in v15.

[DETRIS v1](https://arxiv.org/html/2607.22446v1), §3.1–3.2 and author project, uses deformable
edge controls, barycentric-frame winding classification and a smooth-min distance window.
It is prior art for changing primitive boundaries/expressiveness. This turn does not reproduce
its implementation or establish its exact subdivision equivalence or view-consistency claims.

All these works constrain a proposed method's novelty. v15's narrower role is to measure
isolated native window redistribution on unchanged real inputs and test a conditional
fixed-slot explanation of the saved scene change. Standard compositing algebra and a local
sorting counterfactual are diagnostics, not by themselves new algorithms or quality evidence.

Workflow skills and attribution follow the already-verified record in
`TGS_APPEARANCE_PRIOR_ART_V13.md` (Kassis et al., 2026,
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)).
