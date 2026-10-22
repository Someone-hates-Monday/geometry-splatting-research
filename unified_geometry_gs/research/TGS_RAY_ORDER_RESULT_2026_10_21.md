# Full-image fixed-checkpoint ray-plane ordering: negative quality result

2026-10-21. Authorized scope: mathematical verification and small fixed-input CUDA
diagnostics only. Four native full-image forwards, no training, optimization, backward,
checkpoint changes, or production-renderer edits. Academic objective remains open.

## Result and implication

Changing only the order to per-pixel plane intersection depth did not consistently improve
either original legacy appearance or corrected physical-centroid appearance. Both
prespecified descriptive engineering-lead conditions failed. Therefore the strong v15
local order mechanism is **not evidence of an image-quality optimization**. Retain this
negative result; do not begin a sorting optimization or a broad parameter search on its basis.

Primary unchanged v11 PNG8 PSNR / SSIM, native center order -> ray-plane order:

- Flowers, physical: 20.096152808 / 0.403947868 -> 19.996593630 / 0.399815092.
  Delta -0.099559178 dB / -0.004132776.
- Room, physical: 29.693259639 / 0.792659752 -> 29.694262988 / 0.791268837.
  Delta +0.001003349 dB / -0.001390915.
- Flowers, legacy: 20.139035639 / 0.406235790 -> 20.047185997 / 0.402077378.
  Delta -0.091849643 dB / -0.004158412.
- Room, legacy: 30.871018193 / 0.802938829 -> 30.712430084 / 0.800139053.
  Delta -0.158588109 dB / -0.002799775.

These baselines use pinned precomputed SH colors; tiny quantization differences versus
old native-SH reported scores are possible. Physical native arrays exactly match v14;
legacy native passes its frozen <=2e-4 original-SH RGB / literal noncolor comparison.
No threshold was relaxed. Secondary clipped-float metrics and channel/patch checks are
retained in experiments/tgs_ray_order_v16a/quality_audit.json.

## Correctness and computational evidence

Six software fixtures passed: reversed two-layer analytic RGB, stable equal-depth tie,
zero hit, one hit, all 257 hits retained, invalid negative depth detected. They check the
software and do not constitute physical-geometry proofs.

Both full views, both appearance conditions, reproduce native RGB, alpha and the last/
median contributor indices **literally** before reordering. The physical native capture
also reproduces all five retained v14 arrays. Original model and camera tensor hashes
remain unchanged. Exact permutation, original ordinal/ID membership, alpha-bit identity,
sorting order and depth validity checks pass across all collected records.

- Flowers: 59,780,030 accepted records; maximum 414/pixel; 1,912,960,960 bytes
  (1.782 GiB) of hit workspace, including permutation audit storage.
- Room: 166,037,449 records; maximum 470/pixel; 5,313,198,368 bytes (4.948 GiB).

These counts include contributions behind the original early stop. No fixed K, truncation,
overflow fallback, or hit dropping. Zero flagged eligible nonfinite/nonpositive plane
depths. Eight independently audited sparse pixel packets contain 1,266 records; all
rank permutations and tie rules pass, with normalized plane residual <=2.254e-16.
CPU MSE/PSNR, reversed-axis full SSIM, and direct nonseparable SSIM patches agree with
the primary implementation for all 16 image/metric-mode conditions.

Workspace figures exclude model, native geometry/cache, planes, offsets and output arrays.
There is no measured latency or efficiency gain claim. This exact-sized segmented heap
sort is an expensive reference, not a practical competitor to published sorting pipelines.

## Mathematical interpretation and limits

For a camera-space plane n·x=d and pixel ray x=z*r with r_z=1, the unique finite depth is
z=d/(n·r), provided n·r!=0. Exact midpoint refinement leaves this plane unchanged.
This justifies a refinement-invariant geometric depth reference in exact arithmetic;
it does not imply refinement-invariant alpha, support, color, or rendered output.
The actual implementation computes planes in FP64 from the saved FP32 vertices and view.
It is numerically verified here, not an interval-certified exact predicate.

The v15 identity for a two-layer swap, incoming_T*a*b*(c-d), establishes a possible
RGB change. It places no sign on the error against an external GT image. Fixed models
were trained under their original rendering/appearance convention, and changing that
convention need not reduce reconstruction loss. The current experiment cannot determine
whether consistent training would help or whether sorting is the main training limitation.
Only existing native tile/support candidates are considered; no clipping, filtering,
normal, appearance-gradient or backward corrections are tested.

Per-pixel depth sorting is prior art; see TGS_WINDOW_PRIOR_WORK_V15.md, including
[2D Triangle Splatting](https://arxiv.org/html/2506.18575v3). This experiment is an
evidence-bounded rejection of a direct fixed-model improvement hypothesis, not a new
sorting algorithm or a publication-ready contribution. Both views were already used in
development; two scenes are not independent confirmation or a basis for pixelwise p-values.

## Failure retained and next decision

Original v16 compiled but could not load because Tensor.data_ptr<long long>() is not an
exported PyTorch specialization in this Linux ABI. No GPU fixture or image forward ran
in that attempt. The original source, plan, build and traceback remain under v16.
Version v16a replaces integer interface types with int64_t, keeps the same math and
gates, freezes new hashes, and runs in its own directory. All three executed jobs
(fixtures, real views, CPU audit) terminate with exit code0; this phase launches no training.

Next research priority is a *training-consistent, refinement-consistent representation*
or a useful conditional error certificate, not faster implementation of this negative
fixed-model baseline. Before requesting any longer training, specify one falsifiable
mechanism, a compatible forward/backward definition, controls separating appearance,
support and order, and a small independent-camera validation criterion. In particular:

1. Derive a computable sufficient bound for the actual subdivision RGB disturbance,
   combining isolated window change and weighted active order crossings; explicitly cover
   early-stop and rounding terms. Existing fixed-slot bound alone is insufficient.
2. Audit the proposed bound for coverage and non-vacuity on frozen independent event/
   camera inputs before designing a split decision rule. A bound that is always large is
   not a useful method; observed correlation is not proof of conservativeness.
3. Only if a mechanism survives these checks, request a narrow consistent-training
   comparison with matched budgets and multiple scenes. Such training is outside this run.

The open median_depth_note.tex was not edited. Figure: experiments/tgs_ray_order_v16a/
full_frame_comparison.{png,pdf}. Reproduction: run_tgs_ray_order_v16a.py (frozen plan,
fixtures, real), audit_tgs_ray_order_v16a.py (CPU), and the versioned provenance manifest.
