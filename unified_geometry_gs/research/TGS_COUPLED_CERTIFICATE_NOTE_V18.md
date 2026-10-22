# Coupled prefix and signed-tail refinement of the fixed-cache certificate

2026-10-21. Mathematical tightening of v17; no novelty, training, GT or speed claim.
Same effective-alpha, nonnegative-color, common-event-color, unchanged-external-layer,
normal FP32 and fixed operation-model premises. No proof of preprocessing/power accuracy.

## Signed tails

Let H be the v17 channelwise full-real-composite bound, L_a,L_b the exact omitted
tails after native stopping, E_a,E_b the conditional endpoint rounding envelopes.
The known identity D_native=D_full-L_a+L_b+e_a-e_b immediately gives

    B_signed = H + |L_a-L_b| + E_a+E_b.

This is always <= the v17 sum-of-tails bound. Exact tail evaluation may require
contributions beyond stopping; the inequality alone is not an efficient algorithm.

## Common rounded prefix

Filter both native executed paths to their accepted (alpha,RGB) inputs. Take the
longest identical prefix, length k, without using output error to choose k. With
the same deterministic per-layer FP32 operations, its rounded state C0,T0 is the
same in both runs. Let t0 be the exact product of(1-alpha) over that prefix.
The capalpha<1 implies t0>0. The exact accepted suffix composites are S_a,S_b.
Thus the ideal continuation from the rounded common state is

    C*_a = C0 + T0*S_a,  C*_b = C0 + T0*S_b,
    D* = (T0/t0)*(D_full-L_a+L_b).

The accumulated common-prefix color error cancels exactly. For suffix length l,
use gamma_(3l)*(C0+T0*S) as a conditional channelwise rounding envelope R;
R=0 for l=0. Counting complement rounding, T products, two RGB products and
remaining additions yields at most2l+1 factors per new positive term and at most
l factors on C0;3l covers both when l>=1. FMA contraction reduces the count.
Normal-range and no-overflow premises are inherited from the checked full native
accepted path. No compiler-general proof is implied by matching recorded traces.

Consequently another valid bound is

    B_prefix = (T0/t0)*(H+|L_a-L_b|) + R_a+R_b.

Use min(B_signed,B_prefix) per channel; this is a deterministic intersection of
valid bounds, not fitting a threshold to outcomes. If the complete accepted
(alpha,RGB) sequences coincide, native RGB is literally identical under the same
operation model: return the stronger exact-zero certificate. Ignore primitive
IDs for this equality because native RGB accumulation does not depend on them.
Contributor ordinals and per-primitive statistics need not coincide.

The common-prefix bound may still be loose: it retains triangle inequalities for
window/order changes, loses their cancellation against tails, and does not fully
couple rounding on shared suffixes. A small bound limits perturbation, not GT error,
loss reduction, training convergence or reconstruction geometry.

## Predeclared checks and scope

Freeze source and this formula before inspecting new v18 outputs. First apply it
as a posthoc development analysis to all64 v17 pairs; retain all zero and nonzero
cases. Then freeze other event/camera/pixel inputs for a separate diagnostic cohort.
Report all three ablations: v17, signed tails, signed tails+prefix+exact-zero.
No new tolerances, quality thresholds, p-values or parameter searches.

The method still requires counterfactual child alpha/order information and exact
or safely bounded suffix quantities. Count needed records and suffix lengths;
do not equate a tighter certificate with a cheaper densification rule. If a full
trial render is still needed per split, this version is an analysis/reference
tool rather than a practical adaptive-density method. Standard compositing error
quantification is prior art (GaussianPOP, arXiv2602.06830); novelty remains open.
