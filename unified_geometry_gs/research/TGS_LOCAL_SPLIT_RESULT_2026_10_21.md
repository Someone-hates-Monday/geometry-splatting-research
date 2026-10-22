# Cached subdivision queries: literal native output with lower query latency

2026-10-21. This phase implements and measures a fixed-checkpoint evaluation query.
It is not a training renderer, a new density policy, or a demonstrated reconstruction
quality improvement. No optimizer, training or backward is used; the research goal
remains open and novelty is not established.

## Problem and implementation

Evaluating one proposed midpoint split by the original implementation preprocesses and
sorts the whole changed scene. The new reference reuses one original scene/camera cache,
preprocesses only four children, removes the parent from old tile lists and stably merges
the child keys. It replays affected tiles and copies original RGB/alpha/path output
elsewhere. Held parent color is used, matching the old v14 H intervention. Support,
window,cap,cutoff,center-depth key,tie order and skip-current stop remain native.

The prototype deliberately uses an unchanged native4-child full forward to obtain
child caches; its full-image allocation/rendering is included in the query measurement.
It is not a preprocessor-only implementation. Geometry cache extraction and output
cloning are also included. Stable merge and locality conditions are documented in
TGS_LOCAL_SPLIT_CORRECTNESS_NOTE_V19.md. The four child/external sort keys can cross;
the algorithm does not put children back in a fixed parent slot.

## Correctness on real fixed inputs

Two public7000 checkpoints, flowers938,101 triangles and room2,100,000, two existing
cameras each, all8 frozen v14 events per camera:32 cases. No success-only selection.
Five initial software fixtures cover stable ties,parent absence,empty-after-removal,
unchanged-region copying and257 retained contributions. The fixtures alone are not
real-scene evidence.

All32 complete native reference captures reproduce the old v14 five-array hashes.
Every defined visible external cache field and every visible child cache field matches
its original/four-child cache literally. All merged tile lists match full native lists
after canonical ID mapping:69,342,124 tile-list entries checked, including ties/empty
tiles. All32 global RGB,alpha,last/median ordinal outputs are bit-identical to full
changed-scene native outputs; all unchanged regions remain literal. An independent CPU
reconstruction from old v14 lossless RGB patches matches every reported local RGB hash.
This validates the new implementation against a fixed reference, not GT improvement.

There are19 affected-region cases and13 zero-affected cases. The largest affected
coverage is1.42774% of its image. Thus this evidence is for local proposed edits; it
does not establish advantage for large-coverage triangles, many simultaneous edits,
updated caches or arbitrary scenes. Correctness phase uses exactly68 native forwards.

## Balanced query timing

After correctness passed, a separate plan froze2 paired warmups and8 measured pairs per
case, exactly4 full-first and4 local-first in a deterministic permutation. All256
measured pairs are retained. Timed phase uses644 native forwards total, including4
baseline setups and all warmups. No added repetitions, outlier removal or speed-based
early stopping. First warmup rechecks global query identity per case outside timing.

Hardware: RTX5090 Laptop GPU, driver616.92, Torch2.12.0.dev20260314+cu128, sm120.
Wall measurements synchronize before/after, retain outputs through completion, and
release them before the other endpoint. CUDA event brackets and peak allocated-memory
increments are also retained. Telemetry changes46->48C and862->1515MHz before/after;
clocks are not locked. Balanced pairs limit ordering drift but do not establish a
device-independent or sustained-power performance claim.

Primary statistic is each case's median paired full/local wall ratio, then a descriptive
median across cases (not a ratio of separately pooled median latencies):

- All32 cases:6.24086x median, range2.53546–15.84872x.
- Affected-region19 cases:4.85784x median, range2.53546–8.04366x.
- Zero-affected13 cases:12.69442x median, range6.14973–15.84872x.
- Flowers16 cases:4.83184x; room16 cases:7.75650x.

These are PREPARED cached single-parent query measurements. Event tensor/full-array
construction,precomputed colors,model loading and original cache setup are outside
paired endpoints for both methods. Native full rendering also produces auxiliary
depth/normal/distortion/per-primitive statistics not produced by the query. Do not
describe the ratio as training speedup or a general replacement renderer result.

## Workspace and cache amortization

For affected cases, median ratio of added peak allocated CUDA workspace full/local is
3.26698x, with group medians314,903,040 and97,274,368 bytes. This is ADDITIONAL workspace
above the retained context, not total-model peak VRAM reduction. Retained original
native cache/output tensors plus precomputed colors cost301–691MB per view in this
prototype, beyond model inputs. Holding this cache is a real memory tradeoff; a normal
full render can run without retaining it for future queries.

Observed original-forward/cache-unpack setup times are13.5954,2.41582,16.7340,6.55397ms
for flowers test/train and room test/train respectively, including first-call effects.
Adding each setup once to the sum of that view's eight measured local event medians
gives an ESTIMATED8-query speed ratio1.10973,3.09276,2.42074,4.04134x. Aggregating the
four such estimates yields2.48646x. These are arithmetic amortization estimates, not
separately timed contiguous batches. Equal8-event-mix estimated strict break-even query
counts are7,2,3,1 respectively. Query use should disclose this setup/amortization.

The cache describes one fixed original scene. Sequentially accepting a split invalidates
it unless it is rebuilt or correctly updated. This phase evaluates alternative proposals
against the same original state; it does not apply32 sequential scene modifications.

## Academic interpretation and next test

This is a reproducible efficiency result for a useful but narrow counterfactual-query
operation. It is stronger evidence than a smaller perturbation bound alone: the real
full reference output is reproduced while measured query latency decreases. It does
not show a better final model, GT score, geometry, optimized training loop or broad
algorithmic novelty. The mathematical/implementation contribution needs a prior-art
and scope assessment before any publication claim.

[GaussianPOP](https://arxiv.org/html/2602.06830) already performs render-once local error
quantification for removal. General stable merging and incremental rendering are known;
[NVIDIA CUB source documentation](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh)
specifies stable radix sorting. We have not exhaustively reviewed incremental rendering
or established that this particular TGS proposal-query operation is new.

Next, freeze stress inputs with larger projected coverage and separate cameras, include
moved child appearance, and measure the crossover where local replay stops being useful.
An eventual density policy must use this query to make better decisions under matched
budgets and test on independent scenes. Until then keep it as a validated research tool;
do not promote it to a quality/geometry result or silently start full training.

## Reproduction

run_tgs_local_split_v19.py: prepare/fixtures/validate. audit_tgs_local_split_v19.py:
reference reconstruction. benchmark_tgs_local_split_v19.py: separately prepare/run timing.
analyze_tgs_local_split_timing_v19.py: balanced-pair audit and setup estimates.
experiments/tgs_local_split_v19 holds all plans, receipts, raw timings and query_cost
PNG/PDF (all pairs retained, setup estimate labeled). No experiment failed or required
a GPU rerun in this phase. The open median_depth_note.tex remains untouched.
