# Next implementation stage: faster kernel, then a training adapter

Current authoritative state: HISTORY_GAP_GPU_RESULT.md and
results/history_gap_gpu_checkpoint_v1.json. Overall research goal is active.
GPU graph V2 and CUDA kernel are frozen by experiment manifests; copy to
new files for further changes. Main entry is history_gap_gpu_v2.Engine,
which inherits the custom Function/engine from history_gap_gpu.py. CUDA
solve returns loss[M], seeds[J,4] (p/rho/E/AMP), info[M,8]. Status1 is
interior,2 tail formula (not necessarily no-root),3 before/zero,4 singular,
negative status failed. No CPU root/adjoint input. Source-preserving graph
and active caps are documented; no full visibility differentiation claim.

Profile V1: numerical loss/seed kernel72.699ms vs fields5.536/raw-back10.979
for891 rays. V2 total79.067ms, correct891 plus original32 and fresh32.
Next safe optimization is one warp per ray with stride over all nodes,
compensated local prefix and DD product combination, signed log-tail
reduction and separate density sides. No fixed192/512 truncation. Preserve
the serial native as correctness baseline. Use new source/binary names.
Compare full raw gradients to the existing independent MP references,
all891/9anomalies/3zero,64synthetic and10special cases. Retain every failure.
Time serial and warp in the same process, alternate order, report full
geometry/loss/raw backward as well as kernel times; no whole-training
speed claim. Do not relax the zero-block gates that V1 failed.

After correctness/cost passes, implement fixed-ray sampled regularizer in
a research training wrapper, with actual shared Gaussian IDs and direct
GPU tensors. The diagnostic Engine currently performs preparation and
metadata conversion; do not incur CPU capture/transfer or cold context
setup per iteration and pretend the89/79ms kernel batch is total cost.
Use existing private CSR/query capture only as a frozen discrete graph;
preserve original renderer outputs and camera order. Explicitly handle
empty contributor rays and numerical failures before optimizer steps;
never turn missing roots into zero reward or apply NaN gradients.

Required before quality claims: zero-weight identical-init/camera control,
source/optimizer/RNG binding checks and step+gradient smoke checks, complete
per-iteration costs, predefined weights/budgets, paired training and GT/
held-out image/capacity evaluation. Keep old failed147-view GT and original
raw/direction failures unchanged. Fixed development loss reductions or new
synthetic confirmation are not geometry success or novelty proof.
