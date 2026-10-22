# Next: original complete training recipe, then a paired pilot

UPDATE 2026-10-16: full30k pair and all16 evaluation stages COMPLETE. GT gate
FAILED, independently recomputed; no training active. Read HISTORY_GAP_PILOT_RESULT.md
and CONTINUATION top. Stop this loss's automatic lambda search/multiscene expansion.
The fresh next question is in TGS_SUBDIVISION_AUDIT_NEXT.md; no result claimed yet.

UPDATE 2026-10-15: the original full-recipe four-arm6-step diagnostic is now
COMPLETE, independently audited. Do not rerun it. Read HISTORY_GAP_FULL_TRAIN_RESULT.md.
The fixed30k pair history_gap_pilot_v1 is RUNNING under exec session7120;
poll that handle and actual manifest/PID. Automatic post-training evaluations
are in its controller. Do not mutate pinned sources while running.
The remainder below is the pre-diagnostic planning record, not outstanding work.

Current state is HISTORY_GAP_ADAPTER_RESULT.md and adapter checkpoint.
Actual live adapter history_gap_live_adapter.LiveAdapter works through
shared GaussianModel Float32 parameters; V3 three-scene smoke+MP scatter,
zero controls and36 RGB-only memory updates complete. Do not rerun them
because chat was interrupted. Runtime proof is not training/GT success.

Important observed baseline compatibility: original PLYs contain only
positive-infinity opacity logits (3431/1908/2443), sigmoid1. They stay
fixed during ordinary updates and their gradients must be exactly0.
Do not sanitize/clamp/drop them. All remaining parameters, all gradients
and Adam state must be finite. The coefficient C1 theorem does not claim
a derivative at infinite raw logits. Full training's own opacity reset/
densification can change the mask/topology; log those explicit operations
rather than applying a permanently stale PLY-index mask.

Next priority is scientific quality assessment, not more kernel polishing.
Use an isolated wrapper around actual train.py and original full losses.
train_solver_branch.py/checkpoint_state.py provide bound optimizer restore,
RNG/camera/filter/densification sidecar state; train_holdout_branch.py
enforces actual42 train/7 test pools and NCC/filter-camera exclusions.
The completed factor_holdout55_v2/common/chkpnt6999.pth plus its .runtime.pth
are a likely strictly train-only starting point; verify actual hashes and
sidecar/source before using. The current49-trained30k PLYs cannot support
new held-out generalization claims. Prefer a real common-state controlled
pilot on scan55 before deciding on longer/multiscene training.

Possible minimal integration: scoped Tensor.backward hook, only when its
caller is the verified ROOT/train.py training frame and self is that
frame's loss. Add weight*adapter loss from its actual gaussians/current
primary camera before invoking the original backward once. Preserve all
other backward calls. Existing parent wrapper optimizer-step hooks can
count/check complete-state binding and smoke termination. Record original
pipeline loss, regularizer and actual summed objective separately, since
train.py's existing display variable otherwise omits the injected term.
Alternative source-copy injection is allowed only with exact executed
source/hash and full parent restore/holdout guards verified. Production
train.py/renderer remain unchanged.

Use independent deterministic camera/sample IDs without consuming train
RNG; for real training sample128 image pixels without the old forced MP
probe inclusions (those were diagnostic). Empty rays retain analytic
constant(log4)^2 and full denominator. Abort invalid numerical status
before optimizer update. The tail status is not a no-root certificate.

A first full-recipe diagnostic can use existing live full CSR path even
though it costs24--42ms capture per smoke update. Include that overhead,
state/filter changes and complete iteration cost; do not claim12ms total.
Only then consider reducing capture overhead (shared primary forward/query
buffers and selected-ray replay), with independent forward/backward parity.
Do not spend another stage optimizing capture before checking whether the
loss is useful in a controlled training pilot.

Before any full pilot fix protocol: identical checkpoint/sidecar/hashes,
actual42/7 separation, source/kernel bindings, disabled and computed-zero
controls, original full NCC/normal objectives, a short finite/binding smoke,
weights/budgets and full cost/capacity/GT/heldout criteria. CUDA repeated
controls show nontrivial gradient variability (scan37 zero group1.05%,
disabled-repeat.678%); use appropriate repeated controls and do not claim
statistical equivalence from the engineering gate. For quality use official
GT and held-out images, not only training loss or same-model evaluator seeds.
