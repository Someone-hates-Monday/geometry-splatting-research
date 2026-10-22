# v14: actual-camera decomposition of real-model split perturbations

2026-10-21. Fixed-input mathematics/CUDA only, no training/optimizer/benchmark/download.
The v13 negative distance result motivates this prospective diagnostic. Do not overwrite
sealed phases. Two already-acquired public7000 models: flowers and room. Use their v10
prespecified test/train camera pair and actual decoded resolution, no GT quality scoring.
These are offline split simulations on actual captured cameras, not captured training events.

Use only existing readonly SOs: archived native9d974 as the precomputed-color renderer;
SH-only4fa625 for the independent native physical-center SH baseline check. Freeze module,
model, camera, helper and script hashes before new rendering. No builds or source mutations.
Appearance uses physical triangle centroid p (the corrected intended SH semantics), not
legacy q=(p_z,p_z,p_z). This defines the comparison; historical weights/training commit and
retraining effects remain unproven. Lower-clamp per-primitive SH color exactly as native.

Per scene, baseline on the two cameras first. Candidate parents require opacity>=.01 and
native radii>24 in at least one of the two cameras. Select8 by smallest SHA256 of
'tgs_split_color_v14/'+scene+'/'+integerID, independent of split/color outcomes. Save the
full eligible ID list, its hash, chosen IDs and input parameters before any split forward.
No outcome-based swaps, visibility filtering, candidate threshold tuning or retry selection.

One event removes one parent and appends the official four world-edge-midpoint children,
copies sigma/opacity/SH, effective inference masks1. All other parameters exactly retained.
For each camera render the same changed geometry with two precomputed-color conditions:
H: all children inherit that camera's parent's physical-center color;
M: children use their own physical-center directions and unchanged copied SH coefficients.
All non-child colors identical to baseline, in both conditions. Four child colors are saved.

With B=original geometry/precomputed baseline, define G=H−B, A=M−H, D=M−B.
G includes coverage/opacity/support/sorting/culling consequences and must not be called pure
window/coverage causation. A isolates color-direction intervention at fixed changed geometry.
D=G+A exactly in real arithmetic; keep signed cross term in squared-error decomposition,
do not force an additive 'percentage caused by color' when terms cancel.

Save full B RGB/7aux/radii/maxblend/paths; changed arrays may be saved as lossless minimal
RGB+7aux patches relative to B, after full-image comparison verifies every outside value
identical. Store full-array SHA for reconstruction. Save child radii/inradius/maxblend,
selected points/coefficients/colors and both conditions' noncolor literal comparisons,
including all7aux,fullradii,maxblend,defined last/median paths,tile count. Check noncolor
byte equality for H/M; if false preserve evidence and withhold isolated attribution.

Controls per camera: precomputed baseline repeat must be literal same; independently render
native SH-only and compare RGB with precomputed physical-center baseline maxabs<=2e-4,
with all noncolor fields literal same. Failure stops interpretation, no threshold changes.
At most12 baseline/control +64 split forwards=76; all32 event-camera cases retained.

Primary diagnostic outcomes: unclipped full-image RGB RMS/MAE and maximum absolute error
of G,A,D; affected pixels above fixed3e-5; alpha/depth change; RGB energy cross term;
optional A/G norm ratio only when denominator>1e-14. Also absolute child color changes.
ROI-normalized diagnostics must be labeled separately; full-image denominators stay3HW.
Predefine 'appearance-material' descriptive case as RMS(A)>RMS(G) AND peak(A)>3e-5;
this is an engineering count for these cases, not an automatic scientific acceptance rule.

Rivals: A can be negligible compared with G; A and G can cancel so preserving parent color
can enlarge D; zero or occluded cases can dominate. No hypothesis gains acceptance merely
by a gate. If changes are tiny or dominated by G, do not expand appearance transport training.
Any positive diagnostic still needs prior-art distinction, runtime derivative checks,
same-budget method baselines and actual image/geometry/efficiency evaluation.

Independently reconstruct all full arrays from baseline/patches, verify their hashes and
recompute outcomes and signed energy identity; audit midpoint parameters and colors against
checkpoint plus an independent SH basis. No pixel-level p-values or scene-level claims.
