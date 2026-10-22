# Shared-ID source-preserving descent diagnostic

2026-10-15, before shared results. Prerequisite: factor raw891/891 and
independent240-digit micro check of the seven original finite-difference
failures. Original raw composite gate remains failed (dense890/891;
original direction35/42). This is a diagnosis of whether a common update
exists, not permission to skip that gate or start training/CUDA.

Objective: equal-weight mean squared history gap over all891 fixed rays,
across the three separate models. Link occurrences by (scan, primitive ID),
assert identical raw12 inputs and filtered-scale values for repeated IDs.
Accumulate raw gradients by shared ID and divide by891 before forming a
direction. Do not optimize each ray independently. Scale S as in the raw
protocol. Separate three normalized negative S-scaled directions: full11,
geometry-only xyz/log-scale/quaternion, and opacity-logit-only. Each has
||S^-1 d||2=1 in its active coordinates. Compare equal coordinate step
norms, not equal optimizer effort or training cost. Frozen filters, cameras,
capture baselines, accepted primitive lists and per-view amplitudes remain.

Zero-step control recomputes every ray loss; bit equality is reported.
Fixed steps1e-4,1e-3,1e-2 for each direction, no line search or selecting
a winner after inspection. For full direction also central differences
h=1e-4,1e-5. Re-solve each ray root from source-derived global brackets.
Any missing finite root invalidates that branch mean; never drop such a
ray or substitute zero loss. Report all per-ray changes and mean, each
scene mean, root statuses, Armijo mean condition and measurable-decrease
threshold max(1e-12,1e-8*meanL). Shared directional last-step gate uses
abs(FD-pred)<=max(1e-9,1e-5*abs(pred)); report its value without changing
the original raw direction verdict. Save shared-ID map/directions and
complete per-ray loss arrays. No parameter file is written to the model.

Measure how many unique IDs repeat, gradient energy by scaled parameter
block, and loss decreases obtainable through opacity alone. An opacity
route can reduce gap without moving geometry: if the effective amplitude
of an earliest peak is at least3/4, H at that peak is at leastlog2,
the median is no later than any peak, hence history gap zero. This is a
coefficient-level nonidentifiability result, not proof that raw capped
opacity can reach that effective amplitude on every real ray. Neither
joint descent nor that negative example proves GT/photometric improvement.
