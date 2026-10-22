# GPU implementation of extended squared history loss

2026-10-15, fixed before GPU results. Retain all existing891 rays/51524
nodes, nine old anomalies, three zero-loss rays, and the two previously
missing shared updates. Compare per-node11 raw gradients against the
existing independent MP chain (whole-ray relativeL2<=1e-5, each xyz/
logscale/quaternion/logit block<=1e-4); loss abs error<=1e-9. All must pass.
All rays remain in denominators, including numerical failures.

GPU receives original captured raw/source/sections and camera metadata,
not CPU/MP roots or coefficient adjoints. A source-preserving Float64
factor graph computes true inverse of the near-rotation matrix, filtered
scales, normalized raw quaternion, actual camera inverse/ray, amplitude
screen/3D-filter compensation and caps. A new CUDA numerical kernel solves
the loss and seeds p/rho/E/A; Torch chains those seeds back to raw11.
Do not edit any production renderer or existing frozen research helper.

Algebraic exits: H(minp)>=log2 implies zero loss; H(maxp)<=log2 implies
exact extended loss(log4-Omega)^2 (including absent roots). Otherwise
the median lies in[minp,maxp] and is solved independently by bisection.
The tail code does not itself assert mathematical root absence. It avoids
an unnecessary far-root computation, while preserving the same loss.
Numerical signs are diagnostic Float64, not certified interval signs.
Use stable scaled signed tails and amplitude/energy prefix decomposition,
compensated prefix products/sums, separate before/after density masses,
and no explicit huge median Jacobian. Handle exact zero-density/zero-loss
as zero gradient. Nonfinite or unresolved states yield explicit failed
status and NaN outputs, never dropped cases or zero reward.

Compile strict Float64 flags (no fast math/FMA contraction; explicit fma
only for compensated products). Record actual CUDA binary/source hashes,
Torch/CUDA/device versions, timing including GPU geometry/loss/raw backward,
and separate transfers/compilation. No whole-training acceleration claim.
Initial serial-per-ray CUDA kernel is a correctness implementation.

Independent confirmation after implementation is fixed:32 synthetic raw
world models seed13018, 1/2/4/8 nodes, before/interior/tail/missing-mass
configurations and nonzero kernel0/.3/1, compared to re-solved MP240
composed directional differences/analytic chain under the same gates.
Supplement with explicit singular/common-peak and separated thin-gap
coefficient tests, preserving the known vanishing-gradient limitation.
If a design/implementation fails, preserve the full failed source/results
before amending. Passing is numerical evidence only; paired training,
geometry/held-out images and novelty review remain required.
