# Fixed-cardinality correction for mean-opacity regularization

Scope: the archived TGS term lambda*mean(abs(effective_opacity)), with sigmoid opacity
nonnegative, N fixed original primitives and k distinct selected parents each replaced
by four children inheriting that parent's effective opacity. Unedited numerical
attributes are fixed. This is elementary ratio algebra, not a novelty claim.

Let S=sum_i o_i, mu=S/N, O_T=sum_{i in T} o_i and |T|=k. In real arithmetic,

    Delta R(T) = lambda*((S+3 O_T)/(N+3k)-S/N)
               = 3 lambda/(N+3k) * sum_{i in T}(o_i-mu).

Thus for a FIXED budget k the corrected unary weight

    r_i^(k) = 3 lambda*(o_i-mu)/(N+3k)

sums exactly to the joint mean-regularizer change. The v27 nonadditivity concerns
adding true one-edit changes, whose denominator is N+3, instead of N+3k. These two
statements are consistent. Their difference is

    Delta R(T)-sum_i Delta R({i})
      = -9 lambda*(k-1)*(O_T-k*mu)/((N+3k)*(N+3)).

If the calibration image SSIM supports are pairwise disjoint under the v27 renderer
conditions, the joint calibration image form plus THIS ONE regularizer is therefore
a sum of corrected candidate weights for fixed k. Camera weights must sum to one
if the same scene-level regularizer is attached to an averaged camera objective.
This does not make the cardinality-constrained conflict-graph optimization trivial,
prove greedy optimality, cover other training terms, or cover variable-cardinality
selection with one unchanged set of weights.

No v29 choices are changed: they were selected on the image form alone. The addendum
records coefficients on their existing true checkpoint effective-opacity tensors.
It is supplementary mathematical/implementation validation, not a newly prespecified
regularizer-aware quality comparison and not actual full training objective gains.

Exact data check: each finite nonnegative IEEE binary32 effective opacity is treated
as its exact rational value with denominator2^149. Group mantissa sums by exponent;
their positive integer partial sums fit exactly in binary64 (<2^53), then use Python
integers and Fractions for the full total and selected sums. This proves identities
on the saved input values without replacing them by CPU sigmoid approximations.
Source tensors are recreated on the original GPU and checked against frozen input
hashes. SymPy independently checks the general identities.

Also measure original and joint torch.abs(opacity).mean() in FP32, without multiplying
by an unknown historical lambda. These observed coefficients may differ from the
exact-value real-arithmetic formula due to reduction and rounding. They do not
establish reproducibility for every execution or training update. The actual historical
lambda, size/normal/distortion configurations and a full training loss remain unknown.
