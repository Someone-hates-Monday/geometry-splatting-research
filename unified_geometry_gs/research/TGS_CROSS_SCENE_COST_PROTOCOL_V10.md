# v10 conditional cross-scene operator cost, frozen before timing

2026-10-20. Run ONLY if the independently audited complete four-camera precision cohort passes.
No optimizer/training/GT. Read-only v9 lean coarse, old v5 lean oracle; historical piecewise SHA
f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852 is different numerics/context.
Do not time failed standard or substitute a camera/scene after outcome inspection.

Exactly flowers7000 then room7000, two prespecified cameras each. Each scene is a distinct phase
of this single unlocked-clock device session, to keep only that scene's input resident for meaningful
memory reporting. Camera indices0/1 flowers,2/3 room. No cross-scene population/seed inference.
Per scene repeat3 times all camera×6 permutations of [oracle,coarse,piecewise], shuffled within each
repeat using one continuing NumPy seed202610213.36 blocks/scene,72 total;216 measured F+B pairs.
Three warmups per method/camera:36 additional pairs. Preserve every scheduled call/outlier;
never rerun merely for clock/temp/tail behavior or favorable outcomes.

Before timing freeze complete schedule, scripts/protocol/precision hashes, module hashes, model/
upstream/camera tensor hashes; check full oracle/coarse outputs against their retained precision
arrays. Verify input hashes after each scene. Telemetry before/after each block. No simultaneous GPU
diagnostics or clock/power changes. API allocations, .item host sync and CPU launch gaps are included;
CUDA event intervals are not pure kernel time. Report forward/backward separately and Torch peak
allocated/reserved/incremental; these are not driver memory or full training memory.

Primary estimator per CAMERA: median of within-block coarse/oracle pair-wall ratio. Report also
every raw call, forward/back wall/events, historical context ratios, order positions and temporal
trends.20,000 paired-block bootstrap draws, NumPy seed202610214, percentile2.5/97.5, loop camera
0..3 then contrasts coarse_oracle,oracle_piecewise,coarse_piecewise then forward_wall_ms,
backward_wall_ms,pair_wall_ms,forward_event_ms,backward_event_ms. Intervals are descriptive only
conditional on this session; temporal dependence is not cured by resampling. No p values.

Engineering screen remains each camera median<=.90 and bootstrap upper<1, plus full audited
precision; overall requires all four. A failure is reported, not converted into scientific success.
Record numerical/quality/novelty/training limits regardless of screen. v9 stump results remain a
different prior session; do not pool its raw clocks with this session or claim comparative model quality.
