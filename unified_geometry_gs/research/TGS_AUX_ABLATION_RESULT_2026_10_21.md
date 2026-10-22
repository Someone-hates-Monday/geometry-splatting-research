# Auxiliary-arithmetic-matched query control: smaller gains and setup limitation

2026-10-21. Fixed-input implementation/cost attribution only. No optimizer, training,
backward, GT or geometry improvement. The broad research objective remains unachieved.

## Control and correctness

An isolated copy of the pinned native forward removes maximum-blend atomic updates and
per-pixel depth/normal/distortion/moment/median-depth arithmetic. It preserves preprocessing,
key duplication/sort,window/alpha/cap/cutoff,T and RGB expressions,stopping,contributor
ordinals,API allocations/shapes and retained cache layout. The exact diff is saved.
Unused auxiliary outputs are zero; original native sources and v19 query binary are
unchanged. The control is forward-only: unchanged backward compiles for ABI compatibility
but is unsupported and never invoked. It is not a production/training renderer or the
fastest conceivable dedicated RGB-only implementation.

Control build flags match -O3,--use_fast_math,sm120. New module SHA256
99d5aaec2a1aaa7e578d9737a2e3708c28d6e10044767d5c76278535b2120820.
All6 old/fresh camera baselines and48 prior proposal cases (32 old plus16 stress) pass:
full-image RGB,alpha,last/median ordinal literally match original, as do all defined
visible geometry fields and tile lists. Original captures reproduce prior archived
references. Both original-child and control-child cached queries match full original
outputs. The4-child original/control images themselves are checked. Auxiliary zeros
are verified. Correctness phase uses204 forward calls, no discarded or relaxed case.

## Four-arm counterbalanced timing

After correctness, freeze all16 stress cases and8 measured four-arm blocks/case,
preceded by2 unmeasured blocks. Arms:

- A: original full changed-scene native rendering.
- B: full changed-scene auxiliary-arithmetic control.
- C: cached query with original native4-child forward.
- D: cached query with control4-child forward.

The primary comparison is B/D. Same prepared event inputs/colors and selected outputs;
both B and D use the auxiliary-arithmetic control. A/B estimates the full-scene auxiliary
effect, and C/D tests that effect within the child-query pipeline. Every local call still
pays a new4-child full forward,unpacking,output clones and replay. API unused allocations
and unchanged preprocessing remain in both backends and are not separately minimized.

A four-treatment Williams design repeated twice balances each arm twice at every
position and each ordered within-block neighbor pair twice. Independent analysis verifies
all orders,128 blocks and512 measured endpoints. No outlier removal,added repeats,route
selection or code changes after timing. First warmup rechecks four-arm output identity.
Timing phase uses644 forward calls including4 original/control cache setups.

Primary descriptive case-median paired B/D ratio:

- All16 cases: median2.06759x,range1.06862–4.54813x.
- Flowers8 cases: median1.15191x,range1.06862–2.03897x.
- Room8 cases: median3.56774x,range2.09621–4.54813x.
- Held-color median2.09714x; moved-color2.05132x (descriptive,not causal color effects).

At maximum23.0038% flowers coverage, B/D is1.06862x H /1.07334x M, only about1.07x;
this small margin is not a general guarantee. All16 case
medians exceed1, but one of128 individual matched blocks has local slower than full.
Raw measurements and telemetry remain; one device/two models do not justify p-values
across pixels,scene-general confidence intervals or deterministic performance claims.

Within the SAME counterbalanced run, original A/C median is2.71179x. A/B median1.31441x:
avoiding auxiliary arithmetic explains part of the earlier apparent query advantage.
C/D median1.00060x: the4-child auxiliary arithmetic contributes little in this prototype.
The remaining B/D advantage is conditional evidence for the complete cached-query design
against this full-scene control, not an isolated theorem about one hardware component.
No profiler decomposition,allocation-only ablation or minimal full RGB renderer is tested.

## Workspace and amortization remain essential

Median B/D ratio of additional peak allocated workspace is3.83977x. This is above current
retained context, not total VRAM saving. The experiment holds both native/control caches
to interleave arms; per-method standalone retained cache+color bytes are separately
recorded. Cache retention and prepared event construction are not free deployment costs.

Matched control-cache setup costs10.19396ms flowers /14.63352ms room. Add once to sums
of8 measured D event medians: flowers23.28337ms versus B16.48147ms, estimated ratio0.70786
(local with setup is slower); room26.49646ms versus B37.98510ms, ratio1.43359. Aggregated
estimate1.09415x. These are arithmetic estimates,not separately timed contiguous batches,
and cold setup comparisons are not randomized causal effects. Equal-mixture estimated
strict break-even is25 queries flowers /5 room. State mutation invalidates a reused cache
unless rebuilt or updated correctly; timings evaluate alternatives against one original.

Thus a broad claim of a major general efficiency breakthrough would be unsupported.
There is reproducible prepared-query improvement, especially for the larger room model,
but short-batch practical benefit is marginal/mixed. This is a research tool and controlled
negative/positive evidence,not a validated density policy or model-quality result.

## Next decision

Stop treating full-image one-proposal latency as sufficient method evidence. A practical
counterfactual density tool needs many proposals per reusable view and usually a scalar
score,not a cloned image for each. Next test batching child preprocessing and reducing
only affected-region score differences,with an output-matched full-scene score reference.
Require exact/reference agreement and measured amortized cost before policy integration.
Do not infer training quality,initiate optimizer search or omit cache setup. Academic
novelty,prior-art differentiation and independent-scene/decision utility remain open.

## Reproduction

build_tgs_aux_ablation_v21.py prepare/build (isolated generated source and diff);
validate_tgs_aux_ablation_v21.py; benchmark_tgs_aux_ablation_v21.py prepare/run;
analyze_tgs_aux_ablation_v21.py. All executed jobs exit0 without GPU rerun or tolerance
relaxation. experiments/tgs_aux_ablation_v21 holds receipts,48 case gates,all raw blocks,
telemetry and aux_attribution PNG/PDF. Baseline sources stay read-only. The open
median_depth_note.tex is untouched. No publication-ready novelty is claimed.
