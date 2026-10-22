# Conditional branch consistency for adaptive window rendering

This note applies standard interval/error analysis to the pinned renderer. It
does not establish novelty or a universal IEEE/libdevice enclosure.

At both real outlier pixels, the exact sorted primitive list and cached window
coefficients agree between v3 adaptive and oracle. CUDA trace_v1 reproduces
RGB, weighted depth, alpha, median and contributor endpoints bitwise. Camera9258
first differs at ordinal338, triangle120791, depth88.77187347:

    adaptive: incoming T=.0001216235323, alpha=.1777907759,
              test T=.00009999999020 -> skip current and stop;
    oracle:   incoming T=.0001216235905, alpha=.1777907312,
              test T=.00010000004113 -> include current.

Camera9322 first differs at ordinal760, triangle199338, depth5.941801071:

    adaptive: incoming T=.0001110012236, alpha=.0991095528,
              test T=.00009999994654 -> skip;
    oracle:   incoming T=.0001110013472, alpha=.0991095528,
              test T=.00010000005568 -> include.

The current alpha is identical in the second example: improving only that
primitive's alpha cannot remove the incoming-prefix transmittance difference.
The extra oracle depth contribution is .001919562830 / .00006536750224;
prefix-weight differences and float accumulation residuals account separately
for the final observed .001919746399 / .00006532669067 differences.
Median contributors are unchanged. This is a verified early-stop explanation,
not an attribution inferred only from whole-image deltas or total counts.

For a conditional reference alpha bracket aO in [a-,a+] and reference incoming
transmittance TO in [L,U], all nonnegative, a sufficient next bracket is

    Lnext = round_down(L * round_down(1-a+)),
    Unext = round_up  (U * round_up  (1-a-)).

Monotonicity of subtraction, positive multiplication and rounding proves
inclusion for both rounded subtraction/multiplication steps; outward operations
also enclose a single rounded exact product. In the tested domain alpha<=.99,
and active transmittances near the stopping threshold are normal finite values.
There is no underflow/overflow argument hidden in this step.

The predicate TO>.5 is fixed throughout the bracket unless L<=.5<U.
The predicate Tnext<eps is fixed unless Lnext<eps<=Unext. These are the exact
strict/non-strict endpoint conditions used in classification.
Induction establishes the same discrete path if every preceding alpha validity
decision agrees and each alpha/reference value lies in its declared bracket.
This theorem is conditional on those window enclosures and consistent helper
evaluation across phases. The empirical power allowance and rounded projection
are not turned into rigorous global enclosures by this compositing theorem.

V4 classifies first, then renders: flagged rays use the double-window path and
cache the flag for backward. The classification pass never commits provisional
maxblend or images. Thus a rejected provisional contributor cannot leave a
stale global max. Double edge sums are given explicit rounding; compiler/helper
behavior still must be validated for the tested build, not asserted universal.
All flagged RGB/aux/contributor endpoints were bitwise equal to the oracle in
the127 cases and both complete real cameras. Unflagged path IDs also agree.

The real two-ray conditional interval demonstration encloses both next-T values
and straddles the original threshold at exactly ordinal338 /760. This is saved
alongside the underlying alpha observations in independent_audit.json.

Hard compositing predicates are discontinuous. For current alpha a and incoming
T0=eps/(1-a), a black prefix can vary incoming T through T0 without earlier
termination. At T just below T0, the kernel drops the current primitive; at T0
it includes weight a*T0. A nonzero channel therefore jumps by a*eps/(1-a)
as the input perturbation tends to zero. A global Lipschitz claim across this
predicate is false. With alpha cap c, the omitted current-plus-tail weight is
bounded by eps/(1-c), not eps, under nonnegative alpha and bounded channels.
For c=.99,eps=1e-4, that conservative weight bound is .01.

The target of v4 is the declared double-window / float-compositing oracle path.
It is not exact-real threshold classification. Four constructed median cases
retain2 CPU64/native median-disagreement pixels each, depth difference2, even
though the two native versions agree. No pixel was deleted or relabeled as an
exact-real success. A different median statistic or stopping semantic is a
separate scientific intervention requiring its own declared experiment.

Cost is not sparse replay alone: the prototype evaluates a classification pass
and a final pass across the full image, totaling about1.9999 times the oracle's
window-evaluation count on the two real cameras. This is not a timing ratio.
Counter removal, phase consistency checks, bounded per-ray deferred contribution
storage or tighter alpha bounds are future cost optimizations, not completed work.
