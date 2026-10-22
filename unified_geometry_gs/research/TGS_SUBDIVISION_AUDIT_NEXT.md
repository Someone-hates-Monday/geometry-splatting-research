# Next independent question: optical consistency of TGS subdivision

## 2026-10-20: capacity-matched maximin candidate rejected

Read `TGS_TOPOLOGY_RESULT_2026_10_20.md`. Sixteen new real-checkpoint events,
four cameras and three four-child rules were tested under the frozen protocol.
The training-view maximin selector failed both gates versus classical
longest-edge refinement; do not advance v1 to training or tune it on these cases.
Next bounded prerequisite is native position/sigma/opacity backward validation
against fixed-input finite differences; only SH-coefficient derivatives have
been validated so far. No full training is authorized by the current scope.

## 2026-10-20: real published checkpoint diagnostics now available

The next input gate below is resolved for one real published model. Read
`TGS_REAL_EVENT_RESULT_2026_10_20.md`: 8 offline simulations, 4 fixed cameras,
3 appearance modes and 96 saved variants audited. These are not captured
training events. Local perturbations exist but global single-event MAE is small.
A real cross-view child-culling witness motivated the sharp projective inradius
bound in `TGS_PERSPECTIVE_INRADIUS_BOUND.md`, checked symbolically and numerically.
Next: prior-art scrutiny of the bound and capacity-matched frozen candidate
tests, without claiming benefits from refusing all splits. Full training remains
outside the latest explicit scope; optimizer/history/true event provenance is missing.

## 2026-10-20: items 1–3 completed within the authorized scope

Read `TGS_VALIDATION_RESULT_2026_10_20.md` and raw evidence in
`experiments/tgs_subdivision_v1/`. Exact examples/integration, a finite-distance
rational-interval certificate, and 18 unchanged-native forward fixtures passed.
A separate degree-one SH direction defect was reproduced and corrected in an
isolated variant, with forward and SH-coefficient derivative checks. These do
not establish training benefits or general renderer/gradient correctness.
Next gate: a provenance-checked real checkpoint/event under
`TGS_REAL_EVENT_PROTOCOL.md`; full training remains outside this authorization.

## 2026-10-18: validation package authorized

The user explicitly resumed mathematical verification and small CUDA diagnostics.
Items 1–3 below are now authorized; earlier analysis-only holds are historical.
Full training is outside this package. Authorization alone is not a result.

## 2026-10-18: perspective extension and bounded first validation package

The current moments draft now includes an existence proof for finite-distance
pinhole cameras with identical fixed isotropic intrinsics. It uses general
projected area ratios and compactness, not the midpoint formula at finite
distance. It does not yet supply an explicit distance or a numerical certificate.
The prior-work update records ETS, DETRIS and Revising Densification; boundary
kernel changes, small-triangle culling and view dependence are established topics.

The following is a concrete proposed execution scope, not work already run.
It requires the user to lift their current analysis-only instruction:

1. Locally verify the existing analytical examples and equations, including
   the 3-4-5 triangle, sigma=1, parent opacity=1/2, the one-view feasibility
   threshold and the two-view affine contradiction. Use exact arithmetic where
   available and independent integration for the continuous moments. Preserve
   assumptions and raw outputs; numerical agreement alone is not a proof.
2. Instantiate the documented fixed-intrinsic camera family at finite distance.
   Preserve actual projected world midpoints and unequal child area ratios.
   Seek separately feasible per-view solutions and a certified obstruction to
   the shared system; an optimizer returning infeasible is not enough without
   independently checked residuals or a separation certificate. If this fails,
   revise the proof before moving to rendering.
3. Prepare an isolated build of the already pinned TGS main/native sources,
   without replacing the working GGGS extension. Compare parent/four-child
   frozen inputs against the documented native window, cap, thresholds and
   culling. Start with constant appearance; record any subsequently introduced
   SH or ordering changes separately. Calibrate floating-point repeatability
   before choosing discrepancy tolerances. This is a renderer diagnostic,
   not a training or real-data improvement claim.

No full training, lambda search or geometry-quality conclusion is part of this
package. Real split-event analysis additionally needs a provenance-checked TGS
checkpoint or captured event; do not label synthetic primitives as such. The
current notes do not establish that this real-event input is available. Only
after mathematical and renderer consistency is established should that separate
input requirement and an event sampling protocol be resolved.

## Multi-view update: a shared opacity correction need not exist

Section 6 of `TGS_SUBDIVISION_MOMENTS.md` derives a restricted two-view
impossibility result. When both projections preserve the midpoint split,
fixed-sigma mass/centroid matching with shared nonnegative child opacities
requires identical labelled side-length ratios. A hand-derived example has
strictly feasible solutions separately in each view, yet no shared solution,
even without an upper opacity bound. The camera assumptions are explicit:
orthographic examples and a pinhole example with different focal aspect ratios;
general tilted perspective with fixed intrinsics is not covered by this proof.

Do not turn this into a blanket claim about real training or novelty. Close
this bounded theoretical step and report it. After the user resumes attempts,
the first gate remains a measurement of actual split-event perturbation and
its source, before any full training or shared-opacity correction search.
No experiment was launched during this analysis phase.

## Further analytic update: first moments and opacity feasibility

`TGS_SUBDIVISION_MOMENTS.md` derives a conditional first-moment formula and an
exact interval for opacity-only mass/centroid matching under screen-midpoint
subdivision. High opacity can make that interval empty. This is an analytic
draft, not an implemented method or experimentally validated outcome. The
screen-midpoint special case must not be applied to general world-midpoint
subdivision under perspective. 2DTS v3 already supplies barycentric kernels and
integral compensation; generic versions of those ideas are not new.
The user's analysis-only constraint continues. No experiment has started.

## Update 2026-10-16: static audit completed; experiments remain on hold

Read `TGS_SUBDIVISION_STATIC_AUDIT.md` first. The earlier 2026-10-13 audit
already pinned the official main/native commits; this has now been verified
against current main and 19 selected original files archived with blob hashes.
No CUDA build or rendering experiment has run. Respect the user's analysis-only
instruction until they resume attempts.

The normalized edge-distance window has a shape-independent continuous alpha
integral `2*opacity*area/((sigma+1)*(sigma+2))`. Disjoint triangulation with copied
opacity/sigma therefore already preserves that integral, while changing the
local field. This follows from a conditional analytic argument, not a runtime
result or novelty claim. A future candidate must address spatial redistribution
or discrete implementation effects, not presume missing total alpha.
Size culling, center-based SH/depth, sampling and optimizer replacement need
separate treatment. The original requirements below are retained as history;
pinning/selected source reading are now complete, impact and novelty are not.

2026-10-16. Not a continuation of the failed history-gap lambda search.
No new method, theorem, quality result or originality claim established.

First pin official triangle-splatting and diff-triangle-rasterization commits,
archive exact source plus license, and reconcile the paper v1 window formula
with the actual forward/clamping/alpha/sorting used by that code. Do not install
into or change the working GGGS CUDA environment before isolated compatibility
checks. Read Triangle Splatting+, MeshSplatting and MCMC/conservative
densification primary works before any novelty statement.

Falsifiable first question: does the official midpoint split preserve rendered
alpha/color for a fixed parent, same vertex plane/appearance and background?
Distinguish (a) area/support union, (b) local window/alpha/transmittance and
(c) preservation of a specified state probability. Failure of (b) does not
by itself refute stationarity of an unspecified Markov chain or prove quality
harm. Paper's area-preservation statement is not disputed by an optical defect.

Use exact triangle/edge coordinates and fixed sigma/opacity, compare the parent
to all four actual children, including internal edge neighborhoods and finite
pixel footprints. Include sigma near opaque limit and actual default; retain
all profiles, image L1/max errors and coverage. Verify actual CUDA against
independent formulas before using synthetic evidence to interpret real scenes.
No threshold or real-model impact result has yet been set or observed.

Only if a repeatable optical perturbation exists, assess an opacity-conserving
partition scheme with the full alpha-composition identity, pixel footprint,
external occluders and per-pixel depth order. A scalar opacity rescale may be
insufficient because child windows vanish at internal edges. Establish this
rigorously rather than presuming a solution. Rendering invariance alone is
not a proof of useful optimization: later real-scene split events, full cost,
training/capacity controls and image/geometry validation remain required.
