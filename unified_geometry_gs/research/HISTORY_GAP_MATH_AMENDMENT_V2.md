# v1 failure and v2 numerical amendment

The first run passed all fixed synthetic cases but stopped at real ray 103:
some strictly positive MP alpha values round to zero in Float64. The naive
b/alpha expression produced NaN; JSON serialization correctly refused NaN.
The failed outputs, traceback and original source ZIP remain in
experiments/history_gap_math_v1. This is an implementation failure, not
evidence against the coefficient-space theorem.

V2 retains all cases and original gates. It evaluates b/alpha as
g/[2(1-alpha*g)], and retains log(alpha) for normalized density weights.
Reported Float64 results therefore have an additional Float64 log-alpha
input, reconstructed from the MP world reference. They are CPU arithmetic
checks, not evidence for an alpha-only input interface, CUDA speed, or a
complete raw-parameter training derivative. Zero-rounded alpha counts are
recorded. Nonfinite gradients are explicitly failed rather than serialized
as nonstandard JSON. No training is launched by this amendment.
