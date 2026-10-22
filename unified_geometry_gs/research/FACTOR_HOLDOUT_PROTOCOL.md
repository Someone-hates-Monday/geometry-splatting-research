# Fresh held-out RGB pilot: factor64 versus legacy64

2026-10-12. Written before this experiment's warmup or model-quality results.
This is a new exploratory RGB question, motivated by the completed all-view
pilot's TRAIN PSNR gain. That pilot **failed its original geometry gate**; its
threshold, data and verdict remain unchanged. No geometry claim is rescued by
this experiment. No full Gram-forward training is included.

## Question, rivals and predictions

Does the audited direct precision-factor backward improve excluded-view RGB
reconstruction when both arms share an exact, newly trained 42-view state?
The candidate combines formula fixes, direct-factor differentiation and FP64
local intermediates. A positive result would not isolate any single component.

Competing explanations: (1) useful optimization change transfers to excluded
views; (2) earlier TRAIN gain was overfitting; (3) increased Gaussian capacity
mediates the gain; (4) CUDA training nondeterminism or this scene accounts for
the difference. Both view splits, per-view changes, model counts and geometry
are reported. One matched pair cannot eliminate explanation (4); it cannot
support a significance claim. Correlated views and evaluator seeds are not
independent training replicates.

## Fixed data and training

- scan55, existing sorted numeric image names 0000..0048; loader `eval=True`,
  standard `llffhold=8`. Test: 0000,0008,0016,0024,0032,0040,0048. Train: other42.
- A **fresh zero-start warmup** uses only those42 RGB images through iteration
  6999. Never load the previous all49-view checkpoint. Existing COLMAP sparse
  points/cameras may use all49 views: this is transductive SfM initialization,
  not a claim that test images never influenced initialization. Scene loads
  test RGB tensors for reporting; they are excluded from training loss,
  NCC neighbor target pools and the camera list passed to the 3D filter.
- This is an excluded-RGB training split on a previously explored scene. All49
  images were used in earlier research, including the motivating TRAIN pilot;
  therefore it is not a fully untouched confirmatory dataset.
- Common warmup: frozen original five-pass control, seed0, 6999 updates. Complete
  checkpoint + optimizer bindings + RNG + filter + densification + camera
  queue sidecar are restored for both arms. Warmup before regularization7000.
- legacy64 then factor64; resolution2, 30k schedule, five-pass roots, batch2,
  original NCC, lambda NCC.6 / geometry.02 / depth-normal0, one NCC view,
  mask threshold.9, appearance3. No M1 router or other experimental extensions.
- Both resume branches get a 20-update finite-gradient smoke first. Exact
  initial-state digest, first camera, first loss scalars and forward config
  must match. Full branches restart from the common6999 state, not from smoke.
- Training runtime guards check actual Scene split, neighbor IDs, every filter
  pool, every RGB/depth training render and actual NCC reference/target cameras.
  Held-out render calls are allowed only in the final iteration30000 report
  with gradients disabled. No quality-driven early stopping or manual choice
  of a favorable checkpoint. No automatic retry/relaunch after failures.

## Fixed evaluation and advancement gate

After both full branches finish, use the same original renderer for 42 TRAIN
and7 TEST images, with raw renderer/background/resolution conventions from
`evaluate_images.py`; no exposure fitting. Primary image metric is mean
global-RGB-MSE PSNR (dB), secondary repository SSIM. Per-channel PSNR is also
reported but is not the primary metric.

Mesh extraction: iteration30000, voxel.002, truncation multiplier4, depth8,
largest cluster1. Align by actual training-camera image ID to the corresponding
official DTU calibration center, never by firstN. Same transform in both arms
must be verified. Three fixed mesh evaluator seeds0/1/2 are sampling repetitions.

Before observing results, the **new RGB pilot advancement gate** requires all:

1. TEST mean PSNR factor64 minus legacy64 >= +0.30dB.
2. TEST mean SSIM difference >= 0.
3. Mean mesh overall factor64/legacy64 <= 1.01 (geometry noninferiority guard
   for this RGB question, not the old >=1% geometry-improvement criterion).
4. Final Gaussian count factor64/legacy64 <= 1.10.

These practical thresholds are not significance tests or a publishability
standard. Report every constituent gate. If this pilot fails, retain its
negative result and do not automatically enlarge this exact RGB study.
Even if it passes, independent warmups/seeds, additional scenes, capacity
controls, component ablations, full-renderer derivative checks and a defensible
related-work comparison remain necessary before broad scientific claims.
Sequential wall time is resource accounting, never a speed comparison.

## Provenance and safety of orchestration

Archive actual source bytes, generated backend payloads and source hashes before
warmup; guard them and input-file hashes at every stage. Atomic manifests record
stage start, PID/argv, return code, completion and artifact hashes. Automatic
post-training evaluation does not depend on a chat reply. Preserve failed
directories, logs, sources and already completed legacy experiments.

Research-planning workflow informed by hypothesis-generation skill; skill
provenance: Kassis et al., Scientific Agent Skills (2026), DOI
10.48550/arXiv.2609.00065. This is workflow provenance, not evidence for the
scientific hypothesis.
