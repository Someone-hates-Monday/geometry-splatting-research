# v25 fixed8-proposal complete candidate-score endpoint timing

2026-10-21. Run only after32 packed RGB/full-hash/score identities and independent CPU
audit pass. Four already observed views,each with the unchanged8 H/M proposals. No new
data,GT-driven selection,optimizer,training,backward or policy/quality inference.

A full_score:each complete edited control scene forward then full-frame v24 FP64 loss.
B cached_image:each4-child control forward plus v19 full-image cached query then same
full-frame FP64 loss. C sequential_packed:each4-child control forward plus dynamic ROI
sync/CPU layout/device transfer/packed replay/v24 FP64 loss. D batch_packed:one32-child
control forward then the same dynamic packed-query pipeline for8 proposals at once.
All return8x3 raw MSE/L1/SSIM changes,against the same provided baseline RGB and GT.
Unneeded native API allocations/bank render count in every applicable endpoint.

Prepare models,GT transfer,proposal states/colors,individual child inputs and static full
image loss layout outside endpoints. Include all child/full native forwards,geometry
unpack,ROI planning,GPU-to-host rectangles,CPU layout,H2D layout/weights,patch writes,
FP64 moments,reductions and temporary allocations in C/D. No free child/ROI cache.
Full-frame layout in A/B is fixed and prepared;this is legitimate because its region
never depends on candidate geometry. List exclusions and precision explicitly.

Measure synchronized wall,Cuda-event bracket and incremental peak allocation with v19
helper. All methods use same v24 FP64 oracle;this is NOT a comparison to native FP32
training or an overall training-speed claim. Per view:2 four-arm warmup blocks,then8
prewritten balanced Williams blocks. Arm labels/per-block orders use one hash-derived
PCG seed,without retries. First warmup checks all returned scalar vectors against
sealed v24 values using its frozen tolerances and zero flags. No outlier rejection,
additional repeats or performance-dependent stopping. Four views produce32 measured
blocks and128 measured endpoints. Control-call budget is derived before execution:

    sum_views [1 baseline + (2 warm +8 measured)*(3*8+1)] =1004.

Persist telemetry before execution,baseline setup S immediately per view,every measured
block and per-view complete receipt. Assert the count derived from the plan at end.
This avoids losing S on a final aggregation assertion as happened in v22. Any failure
is retained and recovered explicitly;never silently overwrite/rerun a live process.

S covers original control forward and cache unpack with model/colors/GT already ready;
record retained original outputs/colors plus GT separately,excluding models/proposals.
Primary ratios are same-block A/D,B/C,C/D,B/D medians. Show all blocks and each view,
not only favorable ones. Explicit conditional estimates F/(S+D) for supplied original
RGB without geometry cache and (S+F)/(S+D) when both require original rendering. These
use same-process observed S and median warm endpoints,NOT measured cold batches.
No confidence across independent scenes or guaranteed general speedup is established.
