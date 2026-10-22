# Phase-consistent arithmetic and non-counter cost validation v5

Frozen before v5 native output,2026-10-20. Retain all127 v4.1 inputs and5184 FD
steps, the two full train cameras and threshold counterexamples. Add the two
previously selected test cameras _DSC9269/_DSC9221, full1245x825, on the same
public stump7000 model. No training, optimizer, GT evaluation or novelty claim.

Lock float window subtraction/products/sums and transmittance operations with
explicit RN intrinsics, alongside already locked double edge sums. Use explicit
float compositing products/FMA for deterministic forward instrumentation parity.
No epsilon, median definition, skip-current rule or mathematical window change.
Compare the new oracle to v4 and retain any numerical changes; do not pretend
an explicitly versioned rounding order is necessarily bitwise identical to v4.

Build4 isolated modules: guard/oracle with window counters, and the same two
without atomic counter operations. Keep and zero the same reserved counter
buffers in all4 to isolate instrumentation removal. This is not a memory-layout
optimization or a minimum-memory implementation. Mark counter modules ineligible
for timing. Shader/backward formulas otherwise remain fixed.

Validation gates: retain CPU64 RGB abs3e-5 and VJP1e-3 (zero group abs1e-6) on
95 original cases; old tie null direction1e-6. The32 near-compositing cases keep
their declared native-oracle target and retain CPU64 disagreements. Guard/oracle
RGB/aux/maxblend abs3e-5, radii/contributor/median IDs identical. Require diagnostic
and non-counter forward RGB/aux/radii/maxblend/flags/contributor buffers bitwise
equal for each mode, and selected VJP groups relative1e-5 (zero max abs1e-6).
All real4 cameras compare complete forward outputs/path IDs and fixed RGB plus
weighted-depth dense VJPs, relative1e-3, all gradients finite. Known outliers must
remain corrected. Old5184 FD branch/support records are rerun for lean modes;
FD numerical error is descriptive, not asserted universally1e-3.

Only after all declared precision gates and independent audit pass, time lean
guard and oracle plus the sealed old piecewise module as historical cost context.
The historical module has a different numerical forward; its speed comparison
is not an equivalence claim or a proof of image quality. Use4 cameras,3 warmups
per combination, all6 variant-order permutations repeated3 times/camera, randomized
block order seed202610206,18 measured calls per combination,216 pairs total.
Use debug=False and the same fixed dense RGB upstream and camera/model hashes.
Record native API wall and CUDA-event forward/backward/pair times, allocations,
telemetry. Includes API allocation/.item host synchronization and host-induced
GPU event gaps; not pure kernel or end-to-end training latency. No instrumented
module timing. Do not lock/change device clocks; disclose drift and single session.

Primary contrast is paired guard/oracle forward+backward wall ratio. Preserve
all samples, order, failures and raw device measurements. Bootstrap paired blocks
within camera, seed202610207; intervals describe this session, not independent
scene/seed replication. Never equate sparse flagged-ray fraction with sparse
total work. Failed precision gates defer timing; no threshold relaxation.
