# v7 actual-ray work profile (not candidate rendering or timing)

2026-10-20. Continues authorized mathematics and bounded CUDA diagnostics; no training.
v5 negative efficiency result is unchanged. v6 edge-only conditional enclosure is the starting
point, not evidence of real-ray rejection fractions. This phase tests that missing applicability.

Freeze source, sealed module/hash, camera order and pixel samples before GPU execution.
Use v5 oracle lean module e8aee33c860f346247a1e48a9fc61f70a7b035ef2ea55b800c5960280ad24ab8
read-only. Four inherited cameras in order train9258/train9322/test9269/test9221. Add one test
camera chosen BEFORE output by minimum SHA256('tgs_real_work_v7/'+img_name), excluding these four.
This is a new camera for this method's diagnostics, not an independent scene/training seed.
All cameras use the same published stump7000 checkpoint, 784056 triangles, SH3,1245x825.
No GT matching, optimizer, densification, timing or quality assertion.

## Input/state and parity

Original renderer executes once per camera. Read its actual sorted point list and tile ranges,
geometry cache, clamped tile rectangles, integer anchors, colors, depth, opacity and sigma.
Preserve those tensors. A separate diagnostic module replays each real pixel along the oracle
window/alpha/T/skip-current path using the unmodified sealed v5 stable_window helper in mode3.
The replay computes all RGB and seven auxiliary outputs with identical explicit RN arithmetic,
last/median ordinals and maxblend. All endpoints MUST be bitwise identical to the same call;
the four inherited cameras must additionally match their frozen v5 oracle outputs.
State views must satisfy alignment checks, sum(rect areas)==native tile instances, IDs in range,
per-tile range endpoints in sorted list bounds, and old double coefficients/radii unchanged.
Input hashes before/after profiling must agree. No diagnostic tensor is supplied to original backward.

## Coarse/compensated model

For each visible primitive compute Bx,By = maximum absolute integer local pixel offset across
its ACTUAL clamped tile footprint, including all in-image pixels of its tiles. Do not substitute
triangle AABB or sample-only footprint. Restrict to abs(B)<=8192, integral finite anchor <2^23,
original c and nonzero hi/lo in [2^-60,2^20], as v6. Invalid input/domain means fallback.
CPU audit uses original bit patterns and exact rational arithmetic, not FTZ comparisons.

For each edge compute hi=RN32(c),lo=RN32(c-hi),Delta=sum(abs(c-hi-lo)*[Bx,By,1]);
M6=sum((abs(hi)+abs(lo))*[Bx,By,1]);M64=sum(abs(c)*[Bx,By,1]);
DeltaHi=sum(abs(c-hi)*[Bx,By,1]). All cache sums/products/conversions round upward.
Cache coarse Efast=max_edges(DeltaHi+2^-22*M6+2^-51*M64).
Compensated per-edge error E=(1+2u)*(u*abs(r)+49u^2*M6+Delta)+2^-51*M64, u=2^-24.
This targets the specified ordered binary64 native plane, not exact original projection.
Max is 1-Lipschitz in sup norm: Ephi=max(Eedges) bounds max-edge value error.
Only positive coarse/compensated certificates reject. Negative sign plus interval-separated
argmax certifies inside with same active edge; ambiguous max or sign means fallback.

Every VISITED pair on the oracle path is counted once into disjoint stages:
coarse outside / compensated outside / inside unique edge / inside ambiguous edge /
sign uncertain / domain fallback. Independently count native outside, below alpha cutoff,
actually INCLUDED contributions and skip-current STOP. Per-ray included K is distinct from
sorted ordinal, which counts outside/cutoff pairs too. Save all per-ray counters, histograms
including zero, quantiles, maxima, and fractions exceeding K=16/32/64/128/256.
No inference from counts to measured speed or local-memory allocation.
Full-frame certificates must not contradict native outside/inside/active-edge decisions.

For inside+unique only, estimate whether a simple Float32 normalized u has analytic window
sensitivity <=1e-6 using .99*sigma*du/(uhat-du). Use cached invradius rounding error and
explicit product rounding, plus native binary64 product allowance. Add **empirical**2e-6
power allowance, retain cap/cutoff margin checks. Record actual alpha discrepancy and branch
disagreements against oracle, especially any accept violating the proposed allowance.
These are conditional feasibility counts on ORACLE traversal, NOT a rendered candidate, a
rigorous power certificate, or evidence of matched compositing/gradients. Never feed these
candidate alphas into replay T or output images in this phase.

## Audit selection and preservation

512 pixels per camera, without replacement, NumPy seed202610209, frozen before outputs.
At each sampled ray choose ordinal 1+uint32(pixel_id*1664525+1013904223)%min(tile_list_len,256).
If zero range or oracle stops first, mark no sample, never resample. Save any reached pair's
triangle ID, coordinate, stage, three compensated values/errors, coarse value/error and alpha data.
Replay all reached samples with Fraction coefficient arithmetic, check enclosures, footprint and
native64 branch; cache audit 512 visible triangles by independent fixed hash rank per camera,
plus 32 largest footprint triangles. Selection formula frozen here; no tolerance tuning.
All actual native/replay parity failures, cache failures, branch or envelope contradictions must
remain in raw outputs and block candidate integration until explained. No deletion/retry for
appearance of results. Corrections versioned with reason; input/output originals preserved.

Promotion: profile feasibility only. Renderer integration still needs127 retained cases,
5184FD, four old full cameras + new holdout RGB/aux/path/VJP, compositing interval decisions,
forward/back cache agreement and actual cost measurement. Existing v5 numerical correctness
is not re-counted as new tests here. Original library, model and open LaTeX remain untouched.
