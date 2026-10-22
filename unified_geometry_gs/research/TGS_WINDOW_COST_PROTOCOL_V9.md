# v9 conditional cost protocol — frozen before any timed native calls

2026-10-20. Run only AFTER independent precision_audit.json permits each candidate. No optimizer,
GT/training/loss/densification or complete training-throughput/FPS claim. Failed candidates are
excluded on their predeclared correctness decision, never on observed latency.

Primary comparison: each eligible LEAN v9 candidate / read-only old v5 native64 oracle, same
published stump7000,7 frozen cameras/1245x825/SH3, identical fixed dense RGB upstream and parameters.
Counter builds excluded. Native API scope includes scratch/output allocation, .item host sync,
CPU gaps and forward+backward; CUDA event intervals include these launch gaps, not pure kernels.
Secondary breakdown forward/backward and Torch peak allocated/reserved (not driver/local-memory
or full training-memory peak). Both original native64 and any candidate have already been validated.

Historical piecewise native module f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852
is included ONLY as context; it has different geometry/tile/numerical forward. It is not a correct
same-model baseline or a claim of GT quality equivalence. Report its ratios even if unfavorable.

If both candidates pass: variants[oracle,coarse,standard,piecewise],4 Williams orders
[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1],each repeated3 times/camera,12 blocks/camera,
12 calls/variant/camera,336 measured native F+B pairs. Positions and all directed adjacent pairs
are balanced within the4-order set. If only one candidate passes: variants[oracle,candidate,piecewise],
all6 permutations repeated3 times/camera,18 blocks/camera,18 calls/variant/camera,378 measured pairs.
If neither passes, STOP without timing. Three warmups per camera/variant regardless of result.
Shuffle frozen blocks using NumPy seed202610211, preserving order inside blocks. Do not change
schedule/repetitions or rerun due clocks/outliers/negative results. All raw calls retained.

Before measured calls verify eligible candidate RGB/aux/radii/maxblend/contributor output against
its own frozen precision output, and original oracle against frozen oracle. Hash points/sigma/
opacity/SH/counts/offsets/RGB-upstream/aux-upstream before and after. Model scratch scaling/density
is overwritten by each forward and is not a trained parameter. Freeze build/source/protocol hashes,
selected decisions, complete order and telemetry before timing. No simultaneous other diagnostic
GPU jobs. Do not alter power/clock/system settings. Record clock/temp/utilization telemetry by block
and preserve temporal trends. Any mismatch/timeout is retained; never restart solely on observation
timeout. A repeat needs a distinct documented technical cause and version, not an attractive result.

Analysis: within-block paired candidate/oracle wall F+B ratios by camera. Primary descriptive
summary median of those paired ratios (not ratio of marginal medians).20,000 paired-block bootstrap
draws,seed202610212,2.5/97.5 percentiles conditional on this ONE unlocked-clock session. No p-values,
independent scene/seed confidence interval or generalization. Also all wall/event medians, all raw
outliers, order-position effects and time trends. Same blocks for historical/context ratios.
No optional stopping based on means, tails or bootstrap ranges.

Predeclared engineering screen (not paper-level significance/goal completion): a candidate's
paired median <=.90 on EVERY7 cameras, and within-session descriptive bootstrap upper endpoint
<1 on every camera, plus all precision gates. Peak memory reported alongside; not hidden by speed.
If this screen fails, report the negative or mixed cost result. Even if it passes, GT/cross-scene,
training quality, broader prior-art/new contribution evaluation and ultimate research goal remain.
