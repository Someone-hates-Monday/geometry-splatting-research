# Stable window v2: bounded mathematical and CUDA verification

Frozen before v2 CUDA execution, 2026-10-20. v1 remains a failed experiment.
This is a numerical/correctness candidate; neither novelty nor quality improvement is established.

Retain all 18 old and 48 stress inputs (including every v1 failure), the rational
tile counterexample, and 12 deterministic rotated/clipped/unequal-positive-depth
holdouts at widths 257, 2047, 8191. Freeze their inputs before native output.
Holdout lattice selection uses only the float64 ideal window, not native results.
Use seed 202610202 for four signed RGB VJPs per holdout, retain all rejected pixels.
Eight old signed stress VJPs remain unchanged. Preserve original auxiliary cases.

Forward changes are explicit: evaluate projection in float64 from original float32
inputs; subtract an integer anchor before storing float32 projected vertices;
evaluate edge equations in local coordinates; use area/perimeter inradius.
Keep the original opacity<.01, world-normal/view and radius/extent preprocessing
rules, except that the projected radius is now computed stably. Tile AABB uses
the three vertices of the homothetic alpha>=1/255 support, signed/clamped floor
on both bounds, exclusive upper floor+1, with a one-pixel outward guard.
The guard is empirical numerical protection, not a universal IEEE proof.
No polygon-line intersection or .01 binning cutoff remains.

Compare direct per-pixel Gamma with edge-cotangent reconstruction under this same
new forward. Require bitwise full RGB/aux/radii equality between these two new
versions. Historical whole-image changes are recorded, not hidden or required zero.
Selected RGB/aux vs independent float64 absolute error <=3e-5;
geometry and scalar/SH VJP relative error <=1e-3 (zero reference max abs <=1e-6).
Retain the old explicit homothety nullspace gate <=1e-6 at an edge tie.
No thresholds relaxed after output. All cases execute; failures retained.

For finite differences report representable perturbations and CPU branch fingerprints.
Additionally decode saved native tile rectangles and compare per-selected-pixel,
per-triangle binning membership at base/plus/minus; report pixel alpha-support
stability separately. Do not call a probe same-branch when either changes.
Check every ideal alpha>=1/255 integer sample in each single-triangle holdout
is inside its native binning rectangle. Record false negatives explicitly.

Performance is disallowed until every specified precision/support gate passes and
an independent audit verifies results. A later timing phase must separately name
its scope and compare both backends sharing this forward. This phase has no timing,
optimizer, parameter update, lambda search or full training. Keep all v1 sources,
protocol, inputs, outputs and evidence manifest untouched.
