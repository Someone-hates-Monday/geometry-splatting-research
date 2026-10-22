# Log-domain stability and selective edge evaluation

These are elementary analysis/error-bound applications, not an established novelty claim.
Let a(u)=min(c,o*u^s), c=.99, o>0,s>0,u>0. Set t=log u.
g(t)=min(c,o*exp(s*t)) is continuous and piecewise differentiable; wherever
its derivative exists it is either0 or s*g(t)<=s*c. Integrating on the
pieces, including across the single cap kink, proves

    |a(u)-a(v)| <= c*s*|log u-log v|.

If |u-uhat|<=d<uhat, then min(u,uhat)>=uhat-d and the mean value theorem gives

    |a(u)-a(uhat)| <= c*s*d/(uhat-d).

This bound needs no power evaluation and applies to every s>0, including
the small-s boundary conditioning diagnosed in v2. It is local in positive u;
if the lower bound reaches zero, this criterion must not certify a fast path.

For cached float edge coefficients nf,bf, double-reference coefficients nd,bd,
and exact integer local pixel coordinates x,y in the tested range, define

    Ei = |ndx-nfx|*|x| + |ndy-nfy|*|y| + |bd-bf|
         + gamma3*(|nfx*x|+|nfy*y|+|bf|) + FTZ allowance,
    Ephi = max_i Ei, gamma3=3u/(1-3u), u=2^-24.

The max map is 1-Lipschitz in infinity norm. The implementation uses the
conservative gamma3 upper bound2^-22, 3*FLT_MIN allowance, and upward rounded
nonnegative float arithmetic. Cached coefficient-error bounds are rounded
outward; 1210 finite stored-coefficient bounds were checked with exact rational
arithmetic in the 95-case corpus. The bound is relative to stored double
coefficients; it does not enclose projection/normal construction error relative
to arbitrary precision real geometry.

With positive inverse radius R and its float cache Rf,

    du <= Ephi*|R| + |phihat|*|R-Rf| + multiplication-rounding allowance.

The adaptive candidate falls back when edge sign is uncertain, positive-u
cannot be proved within this scope, the log bound exceeds1e-6, or the computed
alpha lies close to cutoff/cap. The2e-6 allowance for the float power evaluation
is empirical. The original one-pixel tile expansion also remains empirical.
Hence this is a bounded tested adaptive scheme, not a universal IEEE certificate.

Double fallback evaluates normals/offsets/inverse radius constructed from the
original double projection; just converting already rounded float coefficients
to double fails12 of95 cases. VJP uses the same window evaluation but retains
float geometry-chain calculations and accumulation, which are validated rather
than assumed exact. Shader, compositing and gradient accumulation are not fully
double in the oracle64 module.

Compositing needs its own error analysis. For exact nonnegative T and alpha,
if |T-Tf|<=ET and |alpha-alphaf|<=Ea, a sufficient next-transmittance bound is

    Enext <= (1-alphaf)*ET + (|Tf|+ET)*Ea,

with additional floating operation error when comparing actual implementations.
If the resulting transmittance interval straddles the early-stop threshold or
the incoming-T median threshold, an alpha bound alone does not certify the same
path. A candidate next step is to replay uncertain rays with the double-window
oracle path and cache that replay choice for backward; this is not implemented
or validated here. No statement that the observed real outliers are exclusively
caused by one branch is made without a per-contributor trace.

The pinned kernel skips the current contributor when T*(1-alpha)<epsilon.
Since alpha<=c, its omitted current-plus-tail transmittance can be bounded only
by T<epsilon/(1-c), rather than epsilon in general. For nonnegative alpha and
bounded positive channel values, the full remaining weighted channel is at most
T*Cmax. This is a worst-case truncation bound, not measured error. Changing to
include-current-before-stopping would change the forward semantics and requires
a separate declared model/rendering experiment; it is not a hidden precision fix.

In the two fixed real cameras, median depth agrees exactly, but one weighted-depth
pixel in each camera differs beyond3e-5, along with5/8 fewer window evaluations.
These are retained in real_independent_audit.json as targets for compositing
trace/interval analysis, not silently treated as zero-error image agreement.
