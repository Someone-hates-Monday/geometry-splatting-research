# v8 isolated compensated renderer — freeze before CUDA outputs

2026-10-20. Authorized scope mathematics and bounded CUDA diagnostics, no training/optimizer.
Goal remains scientific quality/efficiency contribution; passing correctness is not that goal's completion.
Use v5 native source snapshot, leave original/archive/library/model/LaTeX unchanged.

## Implementation and invariants

Keep v5 projection, culling, tile bounds, original float geometry/normal caches and Jacobian,
explicit RN compositing, two-pass T-interval classification/replay, skip-current stop semantics,
native .99 cap,1/255 cutoff and first-max convention. A shared helper is used in BOTH forward/backward.
Change only window evaluation/precomputed conditional cache. No Gamma contraction or fused rendering yet.

The helper uses actual clamped tile footprint coarse FP32 rejection; hi/lo Dot2 plane computation;
outward max interval [max(r-e),max(r+e)]; strict active-edge separation; active edge error for normalized
u. Reference and hypotheses inherited from v6/v7 notes, not a novel general floating-point theorem.
Domain retains integer anchors/coords, image sizes<=8192, |local coords|<=8192, nonzero coefficients
and hi/lo in[2^-60,2^20]. Add invradius in[2^-20,2^20] for normalization no-underflow assumptions.
Invalid domain or uncertain max/sign uses the ordered native64 window. Phi interval targets stored
binary64 coefficients, not a universal projected-geometry enclosure.
Keep <=1e-6 analytic alpha sensitivity budget and explicitly EMPIRICAL2e-6 power allowance.
Use downward-rounded cutoff distance. Capped/near-cap values always use native64 rather than trying
to certify raw saturation from a bound on the capped alpha. Power enclosure remains conditional.
All integer local coordinates originate in native pixel/anchor construction, not FTZ equal-zero validation.

GeometryState stable_error expands from10 to48floats/triangle: first10 old errors unchanged;
next18 hi/lo plane floats, next16 bound/normalization cache, next2 footprint extents, domain+reserved.
No original normals/offsets repurposed. Adds152B/triangle (=119176512B at784056) to that buffer,
before alignment, not a full training-memory peak. Existing double80B and counters remain.
This is a deliberately explicit correctness layout, not a memory optimization claim.
Compile guard_diag/oracle_diag/guard/oracle, same buffers, WINDOW_COUNTERS1/1/0/0; no timing yet.
Oracle twin must reproduce the old v5 native64 forward, regardless of added allocation.

## Frozen data / tests

ALL127 retained v5 inputs byte-for-byte and same references/gates, including32 compositing cases
and small-sigma boundary counterexamples; no culling of failures.127×four full images and retained
selected VJPs, inherited5184FD records on lean guard/oracle, independent full barycentric reference.
Native RGB/aux/maxblend tolerance3e-5, radii and last/median ordinals identical guard/oracle;
selected gradient gates unchanged (relative1e-3 with inherited absolute-small handling).
Counter twins must be bitwise identical for all forward/cache/path outputs, VJPs agree within1e-5.
FD support stability is separate from numeric FD error; retain all values and do not claim all1e-3.
Preserve prior CPU64/native median-threshold disagreement on four new synthetic fixtures.

Real full scenes: inherited5 cameras9258/9322/9269/9221/9213, plus one unused test camera selected
before outputs by SHA256('tgs_compensated_renderer_v8/'+name) minimum excluding those5.
Same stump7000,SH3,1245x825. New camera is a method holdout, not another scene or training seed.
RGB and weighted-depth fixed VJPs, all6 gradient arrays finite, relative1e-3 gates.
Full RGB/7aux/maxblend tolerances3e-5, native path/radii identical, flagged endpoints bitwise oracle.
Regression pixels9258(116,742),9322(740,268) retain these output/path gates. Flag count may change
with a different approximation/interval; demanding old flags themselves would prevent legitimate
tighter certificates. No requirement relaxed for RGB/aux/IDs or VJP. Record their actual flags/errors.
Compare oracle twin RGB/aux/radii/coeffs to frozen v5(first4) and v7(9213), and a read-only v5 call
on new camera. No GT identity guesses or quality claims.

## Audit and decision

Freeze source/helper/generator/input/camera selection and manifests before native output; preserve
every failed attempt, gate and terminal handle. Verify original452 source hashes and prior sealed
manifests. Audit expanded layout, cache exact rational selected/all small fixtures, outside/active
intervals, explicit source arithmetic and SASS counter removal, selected/barycentric VJP and FD.
Only after all precision gates pass may a separate predeclared cost comparison begin. Existing
v5 negative cost result remains authoritative and cannot be replaced by window-count predictions.
Two traversals and extra cache memory are costs to report; single-pass capacity128/maxblend
deferred commit remains a future candidate, not silently folded into this correctness phase.
Full mathematical power certificate, GT/cross-scene quality, training gain and novelty unproven.
