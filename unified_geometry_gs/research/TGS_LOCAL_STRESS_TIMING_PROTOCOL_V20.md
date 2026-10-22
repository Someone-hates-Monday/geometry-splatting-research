# v20 timing after separate-camera stress correctness

2026-10-21. All16 held/moved appearance cases retained. Two paired warmups, then8 measured
pairs/case; exactly4 full-first and4 local-first in fixed SHA256/PCG order. Cap322 native
forwards (2 baseline cache setups plus16*(2+8)*2 endpoints). No new CUDA code, routing,
adaptive fallback, outlier rejection, extra repeats or speed-based stopping.

Local endpoint includes a fresh native4-child forward with that case's H/M colors,
cache unpacking, complete output clones, mask and replay. It cannot reuse child caches
from correctness or another appearance for free. Full endpoint is the unchanged native
whole-scene split. Both use prepared event tensors/colors outside paired endpoints;
baseline-cache setup and storage are recorded separately. No GT/geometry/training claim.

Reuse the exact v19 synchronization, CUDA-event bracket, wall clock and peak-increment
measurement helper. Outputs live through stop synchronization then are released before
the other endpoint. First paired warmup rechecks global RGB/alpha/path identity.
Record optional before/after GPU telemetry; do not lock clocks or reroute slow cases.

Primary: per-case median paired full/local wall ratio; ratio<1 means local is slower.
Report every case, grouping by scene,appearance and coverage rank fixed before timing.
Assess static candidate*pixel work fraction as descriptive metadata, not measured cycles
or a fitted policy. Add once-observed baseline setup to sums of8 event medians per view
only as an arithmetic amortization estimate; this is not a separately timed batch.
Two models/two cameras do not support general speed guarantees or p-values across pixels.
