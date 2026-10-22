# Three-pixel follow-up, after isolated-window observations

2026-10-21. Primary64 isolated forwards completed; CPU audit found3 empirical conditional
bound violations on room/train camera1, events0 and5. Freeze ALL3 pixels and raw values in
violation_pixels.json, before detailed tracing. Post-hoc mechanism investigation, not holdout.

Additional cap:3 full-scene readonly recaptures (shared original, after event0, after event5),
using original9d974 module, same v14 physical precomputed colors with children held at parent
colors. Every recaptured RGB/7aux/radii/maxblend/path hash must equal saved v14 output before
interpretation. Extract actual tile-sorted candidate IDs and visible primitive window cache
for each frozen pixel. No uninitialized invisible cache reads. Save packets before replay.

Build one isolated tiny CUDA trace observer, existing CUDA12.8/sm120/gcc11, O3/use_fast_math
matching native receipt; do not change production or original modules. Freeze observer source,
script, module receipts, input/audit/source hashes before compilation and recapture. Observer
mode0 replays exactly the original dot/pow/cap/cutoff/skip-current stop and RGB accumulation.
Require literal RGB/alpha/last-ordinal endpoint equality at all6 before/after pixel endpoints.
If not, retain and withhold mechanism conclusions; do not relax to a numeric tolerance.

After endpoints pass, compare a fixed2x2 on changed packets: actual order versus relocating
candidate children as one group to the old parent's tile-list slot, and original stop versus
no early stop. Alpha cap/cutoff stay unchanged. Check old external candidate IDs/order exactly
match changed external candidates after canonical ID mapping. Color remains inherited.
These are diagnostic counterfactuals; they do not implement a new method or imply pixel-correct
depth sorting. Save complete evaluated candidate traces, packets and all endpoints. Before
packets additionally receive no-stop replay. At most18 tiny trace invocations, no optimizer,
backward, training, clock benchmark, broad camera search or outcome-based pixel replacement.
