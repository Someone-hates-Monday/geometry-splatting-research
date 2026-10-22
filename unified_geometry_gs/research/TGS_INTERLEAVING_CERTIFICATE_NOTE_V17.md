# Conditional subdivision RGB certificate with interleaving

2026-10-21. A derivation from standard alpha compositing, not a novelty claim.
No claim that a small perturbation improves GT loss or subsequent training.

## Exact arithmetic statement

Fix one pixel. External layers j=0,...,N-1 retain their relative order, alpha b_j
and RGB d_j. All alpha are in [0,1], colors are finite nonnegative, and background
is black. Let U_j=product_{k<j}(1-b_k). The old event is a parent of alpha P and
color c, inserted before external m. Its children have effective alpha a_i and
the SAME color c; their actual positions may interleave with the external layers.
Effective alpha is zero for native support/cutoff rejection. Let

    R = product_i(1-a_i), Q=1-R,
    V_j = product_{children actually preceding external j}(1-a_i),
    V*_j = 1 if j<m, else R.

Let B_m be the external-only suffix composited from m to N with black background.
Then the exact full-composite change, children minus parent, is

    D = (Q-P)*U_m*(c-B_m)
        + sum_j U_j*b_j*(d_j-c)*(V_j-V*_j).                 (1)

Proof: for any interleaving, subtract constant c from every layer and background.
Children then have color zero. The external residual contribution is
sum_j U_j V_j b_j(d_j-c), and the background residual is U_N R*(0-c).
The constant c and final background residual cancel between actual and grouped
child orders. This yields the sum term. Grouped children versus parent have
the ordinary fixed-slot replacement difference (Q-P)U_m(c-B_m). Add both.
This proof covers zero/opaque alphas without division by a transmittance.

The per-channel bound is

    |D_ch| <= |Q-P|*U_m*|c_ch-B_m,ch|
              + sum_j U_j*b_j*|d_j,ch-c_ch|*|V_j-V*_j|.    (2)

Only external layers with different child prefix products contribute. This sum
can be computed in O(N+#children) with a prefix scan, not by enumerating every
adjacent swap. It is conditional on unchanged external effective layers and
equal event color. It is a bound, not an exact magnitude: signed cancellation
can make it loose. No bound on quality improvement follows.

## Native early-stop and rounding

Treat the CUDA-computed effective alphas as exact inputs. Equation(1) does not
certify errors in __powf, preprocessing, SH evaluation, clipping, or projection.
For each native endpoint, let S be the accepted prefix before the skip-current
test stops; let F be its full exact composite, F_S its exact prefix composite,
and L=F-F_S the omitted nonnegative exact RGB tail. Thus

    D_native = D - L_after + L_before + e_after - e_before.

A conservative channel bound adds L_after+L_before and bounds on |e| to (2).
The tails must include the skipped current layer, not just layers behind it.
The stop condition test_T<1e-4 alone DOES NOT imply omitted RGB<=1e-4*Cmax:
with alpha capped0.99, incoming T may approach0.01. Use the actual prefix/tail.

For an accepted prefix of n layers, assume normal FP32 nonzero intermediates,
no overflow, round-to-nearest and at most two products plus one accumulation
per RGB contribution, and subtraction+product per T update. A standard positive
sum/product forward-error bound is |e_ch|<=gamma_(3n+3)*F_S,ch,
gamma_k=ku/(1-ku), u=2^-24, k*u<1. Counting each factored term's rounded complements,
T products, two RGB products and remaining additions gives at most2n+1 factors;
3n+3 is conservative and also covers contraction into FMA. No cancellation is
used since native RGB colors are nonnegative. A data audit must check normal-range
and overflow premises; otherwise this version returns 'not certified'. This is
conditional numerical analysis, not a complete proof of CUDA compiler semantics.

## Scope, prior art, and falsification

[GaussianPOP](https://arxiv.org/html/2602.06830) Section3.2 already derives the
fixed-slot single-deletion difference, and Section3.3 uses local cumulative
quantities. Fixed-slot identities and render-once local evaluation are not new.
[SteepGS](https://vita-group.github.io/SteepGS/) studies loss-based Gaussian
splitting with a second-order model. Neither a new densification method nor
novelty of (1) is established by this note; a targeted search is not exhaustive.

First check (1) in exact rational arithmetic on all three old v15 development
pixels, including their actual recorded alphas. Then freeze a separate real-input
validation cohort and test premise coverage, actual error containment, and the
bound/error ratio including zero cases, without retuning a tolerance. A method
proposal requires bounds tight enough to make useful split decisions at acceptable
cost, independent views and eventually a training-consistent comparison.
