# v16 full-image ray-plane ordering baseline

2026-10-21. Fixed-input CUDA/math only; no optimizer, retraining, backward or production
source changes. Previous v15 is sealed. This is a reference baseline for a known sorting
idea, not a new algorithm. No training or practical efficiency claim follows from this test.

Use existing public flowers/room7000 models, their v10/v14 selected test camera0 and exact
resolutions, verified v11 GT identity (no crop/resize/exposure adjustments). Both original
camera views have already been used for development; not independent confirmation. Two
appearance conditions: physical-centroid p and legacy q=(p_z,p_z,p_z), each evaluated with
the same pinned polynomial and native lower clamp. Keep every condition and outcome.

Keep native preprocessed support/window/alpha/cap/cutoff exactly. Extract original visible
cache and tile candidate lists. Collect EVERY alpha>=1/255 candidate at each pixel, including
ones behind the original early stop. Count then prefix-scan and allocate exact-sized storage;
no fixed K, truncation or overflow fallback. Maximum new hit workspace8GiB per view; exceeding
this blocks that view, not permission to drop candidates. Count all nonfinite/nonpositive
ray depths; any such eligible hit blocks 'physical ray ordering' quality interpretation.

Compute per-triangle plane in camera space using FP64 applied to saved FP32 world points
and view matrix, ray direction ((2x+1)/W-1)*tanFovX,((2y+1)/H-1)*tanFovY,1. Use FP64 ray-plane
z for sorting, original candidate ordinal as a stable exact-depth tie break. Sorting changes
order only; no alpha/support/color changes. It does not correct near clipping, screen-space
window, normals, losses, backward or ray sorting visibility outside the native tile lists.

Implement a segmented heapsort over full per-pixel hit arrays, preserve original accepted
rank and ordinal. Verify exact permutation via a separate original-ordinal map and per-source
rank atomic seen table, plus ID membership, alpha bits and sorted-key order checks. Any
invalid or duplicate/missing record blocks scoring. It is a diagnostic reference, not an
optimized competitor to published local resorting pipelines.

Prerequisites before real photos: deterministic2-layer reverse-depth case with analytic
RGB, exact-depth tie stability, no hits, one hit, and257 accepted hits (no cap). Include an
invalid-depth fixture that must be flagged. Freeze all source/protocol/helper/model/GT hashes
before running fixtures or new image outputs.

Real-view gates before GT scoring: center-order replay from collected hits must literally
match native RGB/alpha/last+median ordinal for each appearance condition and full image.
Physical-p native must reproduce v14 saved baseline array hashes. Legacy precomputed native
vs unchanged original-SH output may differ through polynomial operation rounding, but must
match <=2e-4 fixed max RGB with noncolor literal. Gate failure is retained without loosening.

At most4 native full forwards: p and q per scene. Original-SH check reuses retained v12
old-posthoc original output, no new forward. One hit collection/sort per scene reused for
both appearance conditions. All images, raw/clipped metrics, zero/invalid/count/permutation
metadata retained. No complete native replay/backward or full renderer speed claim.

Primary quality: unchanged v11 PNG8 MSE/PSNR and fixed SSIM (11x11 sigma1.5 zero padding5,
RGB mean); secondary clipped-float metrics. Scoring only after all prerequisites and both
scene gates pass. Need an independent metric and sparse-order audit. No p-values across
pixels/two scenes. An engineering lead is >=.1dB PSNR improvement on BOTH scenes with
nondecreasing SSIM within an appearance condition; this is not a scientific acceptance rule
or a claim of training benefit. Retain all failures and the legacy-vs-p interaction.

Cost is secondary and only after correctness. This version first records exact hit count,
workspace and algorithmic pass counts. If later timing is added, freeze balanced repeats
separately and label cached replay versus end-to-end native pipeline. Do not compare an
unoptimized reference's extra preprocessing/replay cost as an optimized method speed result.
