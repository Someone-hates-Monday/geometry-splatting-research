# Exploratory history-gap geometry pilot, fixed before full training

2026-10-15. Entry gate: original full-recipe smoke v1 completed four independent
six-update runs, with all computed-zero first-gradient groups inside the
predefined disabled-repeat engineering bound. This does not establish quality.

Question: does weight .01 extended squared history-gap regularization improve
official DTU scan55 geometry under the original GGGS recipe without material
image/capacity degradation? Rivals include optical mass collapse, poor transfer
from local loss to geometry, adaptive capacity and CUDA trajectory variation.

Two fresh processes in fixed order control, history_gap. Both restore the
exact strictly42-trained common6999 checkpoint AND runtime sidecar verified in
results/history_gap_full_train_preflight_v1.json. They run original control
renderer/train.py through iteration30000 (23000 further Adam calls; final30000
has backward/report/save but no Adam). Use the full recipe and128 independent
uniform sample pixels specified by HISTORY_GAP_FULL_TRAIN_PROTOCOL.md.
Weights disabled vs .01. Normal/NCC/geo, densification/reset/pruning/filter,
camera queue, optimizer binding and schedule stay original. No test-driven
checkpoint selection, tuning or early stopping. Runtime guards exclude all7
test images from train render/NCC/filter; final no-grad report alone allowed.

Sampling uses a local NumPy generator keyed13023, iteration, primary camera.
Fixed current visibility/filter/contributor history is differentiated locally;
this is not a derivative through the discrete renderer or topology decisions.
Keep existing +Inf opacity as fixed opaque coordinates under ordinary Adam,
check all other parameters/grads/Adam finite every update and reject invalid
native status before update. Log original explicit topology/reset operations
via before-backward/at-optimizer counts. Every update goes into JSONL; compact
progress is written each100 updates plus the first6. First-gradient artifact
cost and full process wall time are recorded. Full CSR and all diagnostic
checking overhead count as implemented cost, with no speedup claim.

After BOTH training processes complete, automatically evaluate:
- mesh_extract.py iteration30000, voxel .002, truncation4, depth8, largest1;
  train42 depth images only, original public renderer;
- align_holdout_mesh.py uses the actual42 numeric camera IDs and official
  camera centers, identical transform required across arms;
- official-style DTU evaluator, fixed sample seeds0/1/2, all three directed
  distances and overall retained; these are evaluation sampling repeats;
- evaluate_images.py all42 TRAIN and7 TEST, raw renderer, no exposure fitting,
  primary global RGB MSE PSNR and repository SSIM, all per-camera rows;
- final Gaussian count/PLY bytes and full training resource cost.

Fixed advancement gate, all required (engineering usefulness, not significance):
1. mean_overall(history_gap)/mean_overall(control) <= .99;
2. each mean directed distance ratio <=1 (no trading accuracy for completeness);
3. TEST PSNR delta >= -.10dB; TEST SSIM delta >= -.001;
4. final Gaussian count ratio <=1.05;
5. sequential full-process training wall ratio <=2.0, including full CSR and
   checks (resource accounting on this run, not a controlled speed benchmark).
All numbers and failed constituents will be retained. No automatic enlargement
or weight search if this fixed pilot fails. A pass permits independent seeds,
additional scenes and mechanism/weight/capacity ablations; it is not publication
success. One shared warmup pair, correlated camera views and evaluator seeds
do not establish statistical confidence or independent training replication.
Scan55/all49 images have prior research exposure, and SfM initialization may
use all views. This is exploratory excluded-RGB training, not pristine validation.

Actual sources, input hashes, binary identity, PID/argv, exit status, checkpoints
and evaluation artifacts are recorded. No automatic rerun after failure.
The new controller performs evaluation without waiting for another chat reply.
