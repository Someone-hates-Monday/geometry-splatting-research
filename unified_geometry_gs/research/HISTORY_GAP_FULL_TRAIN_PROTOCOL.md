# Original full-recipe integration diagnostic, fixed before execution

2026-10-15. This is an exploratory integration gate, not a quality study.
The intervention is sampled extended squared history-gap loss, weight .01,
128 uniformly sampled image pixels per primary training camera and iteration.
Empty rays retain their analytic constant and the full sample denominator.
Independent sampling seed 13023 plus iteration and camera integer consumes no
training RNG. No forced MP probe pixels. Compute-zero weight is exactly 0.

Four separate processes: disabled_a, disabled_b, computed_zero, positive.
All resume scan55 strict42 train/7 test common6999 checkpoint AND runtime sidecar;
canonical restored state must be identical, optimizer bindings verified.
Recipe: original control renderer/train.py, resolution2, DTU, eval=True,
decoupled appearance3, 30000 schedule, regularization starts7000,
lambda_depth_normal0, NCC .6, geometry .02, one neighbor, threshold .9.
All original densification/pruning/opacity reset/filter calls remain enabled.
Each arm intentionally stops after six actual Adam calls (7000..7005).
Parent wrappers therefore record DiagnosticComplete as an exception; only the
outer wrapper may classify this exact exception as intentional, after checking
actual resume, depth/NCC, filter and train/test camera guards.

Before the first densification save full parameter-group gradients from the
first backward. Per group compare computed-zero vs disabled_a relative L2
against max(1e-5,5*disabled_b-vs-disabled_a). A None mismatch fails; exactly
zero reference norm uses denominator1e-30. This is an engineering noise gate,
not equivalence testing or a statistical significance claim. Camera and
sampling IDs must match across applicable arms. Later trajectories/topologies
may diverge through CUDA atomics; report them, do not freeze topology or
silently align unequal final shapes. A positive-weight result cannot establish
held-out/GT gains from its own loss reduction.

All grads, ordinary parameters and Adam state must be finite. Existing raw
opacity +Inf is retained as a fixed opaque coordinate during each ordinary
Adam call, with exactly zero gradient. Derive the mask from the CURRENT
topology each time; explicit original reset/pruning operations may change it.
Log counts before backward and around Adam. Abort new nonfinite values,
nonzero opaque gradient or invalid native status before optimizer update.
Record baseline RGB/NCC/geo losses separately from the injected objective.
Full cost includes query/count, scan/allocation, CSR capture, loss construction,
backward, original topology/filter operations, optimizer and finite checks.
First-gradient artifact transfer/write and checkpoint verification are timed
separately. No whole-training speedup claim is made.

After this gate, fix a paired full training pilot using this same initial
state and full recipe. Evaluate seven excluded images, official DTU geometry,
primitive count and complete cost. No tuning to the test views. Scan55 has
already been used for research and is exploratory, not a pristine validation
scene. Multi-scene independent training replication remains required.
