# Fixed-code query stress: coverage23%, moved appearance, setup caveat

2026-10-21. Reuses the sealed v19 query CUDA binary unchanged. No new CUDA build,
optimizer, training, backward, GT comparison or policy/routing search. The research
objective remains open. This phase extends correctness and measures limits of a
cached counterfactual tool; it does not establish a new density policy or novelty.

## Inputs and output coverage

The other existing cameras _DSC9136 flowers and DSCF4755 room were not v19 query
validation/timing views. These are known public7000 models, not independent new scenes.
Before any child trial, select4 eligible parents per view using only original native
tile-rectangle areas, targeting1,0.5,0.1,0.01 of that view's maximum eligible area,
with nearest target and smaller-ID tie breaking. Eligibility: radius>0,opacity>=0.01.
The full eligible data and selections are retained; independent Python min-key selection
reproduces all8 IDs. No observed split output or timing selects a parent.

Affected output coverage for flowers is23.0038%,13.2189%,2.16622%,0.221545%; room
11.2939%,6.87454%,1.38501%,0.110880%. This genuinely exceeds the old1.42774% maximum,
but does not test full-frame/many-simultaneous edits or establish a crossover point.

Both held-parent RGB H and separately evaluated child-centroid SH RGB M are tested
for each parent:16 cases. All global RGB/alpha/last+median ordinal outputs match
complete native split renders bit-for-bit. External and child visible cache fields,
all merged tile lists and outside-region outputs pass the unchanged v19 gates.
Native H/M noncolor arrays are literal. Baseline all5 array hashes reproduce v18
fresh captures. Save each actual query as a lossless baseline-plus-bounding-patch;
CPU reconstruction independently reproduces all three output hashes versus native.
Correctness uses exactly26 native forwards. No selected case fails or is discarded.

## Fixed balanced timing

Only after correctness, freeze2 paired warmups and8 measured pairs/case with4 full-first
and4 local-first.128 measured pairs retained;322 native forwards including original
setups/warmups. Reuse unchanged v19 timer/synchronization/peak-increment helper. Local
endpoints pay a NEW native4-child full forward for their own H/M colors, unpacking,
output copies and replay; they cannot reuse a correctness/other-color child cache free.
Event arrays/colors are prepared outside endpoints for both methods. No fallback or
new code is introduced after seeing a slow/fast case.

All16 case median paired full/local wall ratios exceed1, with descriptive median2.63791x,
range1.30765–6.26105x. Flowers8-case median1.43392x; room4.91728x. Held-color median2.68546x,
moved-color2.58674x. These condition medians are descriptive, not causal appearance
speed comparisons: overall case order is not randomized across appearances/views.

At maximum flowers coverage23.0038%, H is1.38603x and M1.30765x. At maximum room
coverage11.2939%, H is2.79702x and M2.70193x. Thus gain shrinks substantially at the
largest tested coverage; no local-slower case or measured crossover is found in this
finite cohort. It would be unjustified to claim that local replay always wins.

Median additional peak-allocated workspace full/local ratio is3.84531x, excluding
retained context; it is not total VRAM saving. Original native+color caches cost
329,140,851 and680,209,695 bytes per fresh view beyond model inputs. Full native produces
depth/normal/distortion/primitive statistics which the query does not produce.

## Setup changes the practical conclusion

Observed once-per-view setup is14.73094ms flowers /16.91653ms room, including first-call
effects. Adding it to sums of8 local event medians gives28.09360ms flowers versus full
21.08977ms, an estimated full/local ratio0.75070: this eight-query flowers mixture is
SLOWER once setup is included. Room is29.77115ms versus full55.43445ms, ratio1.86202.
Combined arithmetic estimate is1.32247x. These are not separately timed contiguous
batches. Equal-mixture estimated strict break-even counts are16 and4 queries respectively.
Do not omit the flowers setup-inclusive counterexample from a speed claim.

The results do not establish complete densification or training speed. Model loading,
event definition/full-array construction, baseline cache invalidation after an accepted
edit and multiview retention are outside prepared-query timings. No clock locking is
used; raw CUDA brackets, wall samples, memory increments and telemetry are retained.
Two fixed models/one device do not justify scene-general guarantees or pixelwise p-values.

## Required control before attributing speed to cache reuse

The old full native renderer computes additional depth/normal/distortion/max-blend
outputs. Current timings measure the useful end-user query versus that implementation,
but do not isolate cache reuse from avoiding those computations. An output-matched
full-scene control is now required: keep preprocessing,duplication/sorting and the same
RGB/alpha/path recurrence, ablate only unused auxiliary shading, validate literal
outputs, then compare with the cached query under a separately frozen protocol.
Retain native allocation/interface costs as a separate disclosed factor. This is the
next priority before improving kernels, fitting a routing threshold, claiming academic
efficiency gains attributable to a mechanism, or considering longer training.

No general novelty of stable merges, incremental rendering or render-once local queries
is established. Prior-art and independent-scene/decision-policy evidence remain needed.
No GT/geometry gain is inferred from matching a counterfactual reference image.

## Reproduction

run_tgs_local_stress_v20.py prepare/run; audit_tgs_local_stress_v20.py; separately
benchmark_tgs_local_stress_v20.py prepare/run; analyze_tgs_local_stress_timing_v20.py.
All jobs terminate exit0 without GPU reruns or parameter retuning. Results, full eligible
selection data, lossless output patches, raw pairs and stress_query_cost PNG/PDF are in
experiments/tgs_local_stress_v20. The figure labels setup estimates. All frozen input
hashes are checked before evidence sealing. The open median_depth_note.tex is untouched.
