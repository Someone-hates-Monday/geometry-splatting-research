# Live GaussianModel adapter smoke protocol

2026-10-15 before live results. Three original frozen30k scenes37/55/63,
camera0000 each, memory-only updates; production sources/PLY unchanged.
This is a fresh-Adam fixed-topology RGB-only integration smoke, not the
full existing NCC/normal training recipe, resumed optimizer or quality run.
Use GaussianModel.training_setup(default OptimizationParams), scene extent
as spatial_lr_scale and update_learning_rate(30000). Record all LRs.

Capture actual per-pixel query/AMP with the pinned actual_query native;
derive counts/offsets and capture complete CSR using the pinned native.
Validate CSR status and check original public renderer RGB/alpha/normal/
depth/slope/radii bits against both private captures. Only sampled rows
are packed to Float64 on GPU. Gathers originate from actual shared
GaussianModel raw Float32 parameters; filters/accepted IDs are fixed within
each local derivative, as in the mathematical model. No full-ray CPU copies
in the adapter path. Small metadata/scalar checks and sample indices are
allowed and their synchronization is included in end-to-end wall cost.

Sampling smoke:128 uniform pixel IDs without replacement using independent
NumPy Generator(seed13021+scan), union the original MP probe pixels for
camera0000. The MP subset is a diagnostic forced inclusion, not a proposed
training sampler. Preserve all sampled empties: loss=(log4)^2 and zero
gradient for empty list (Omega0), counted in the full sampled denominator.
The native rejects empty lists, so bypass only their analytic constant,
not their denominator. All-empty handling must be explicitly exercised.

First check source/raw/ID/sections for every original MP probe against
the frozen world_root_vjp_v1. Compare actual Float32 shared parameter
scatter for reg-only mean against summed MP raw gradients divided by
ray count: relative L2<=1e-5 and each of xyz/logscale/quaternion/logit<=1e-4.
Check other parameter groups have no regularizer gradient. Audit full
sample counts, empties, IDs and all finite/nonnegative statuses; abort
before optimizer step on any failed/nonfinite numerical state.

Zero-weight test: from unchanged parameters compute two RGB-only control
gradients, then RGB+0*live_regularizer gradients. RGB loss uses default
lambda_dssim L1/SSIM against actual camera RGB. Each group error versus
first control must be<=max(1e-5,5*second-control error); None gradients must
match. Record existing atomic nondeterminism; do not attribute it to loss.
Capture/adapter must leave Python/NumPy/Torch CPU/CUDA RNG states unchanged.

Short updates: snapshot actual parameters and fresh Adam state, then
restore the same objects/state for two disabled3-step repeats, computed-zero3 steps,
lambda=.01 three steps; fixed camera/sample list, no densification/filter
updates. Check optimizer parameter-object bindings, all gradients/updated
parameters finite and unchanged number of Gaussians. Compare disabled/zero
parameter states with relative<=max(1e-5,5*disabled-repeat deviation);
this is a nondeterminism-aware engineering control, not bit determinism
or statistical equivalence. Report all changes for nonzero without
requiring decrease or calling this training/GT success. Save small selected
parameter/gradient arrays, all counters/times and state digests. All scene
input PLY hashes rechecked. Include public render/backward, private capture,
CSR scan/validation, sampling/packing/prepare, loss/backward/optimizer and
total synchronized wall cost, cold observations retained separately.
