# Conditional outside certificate for the v9 native64 reference

2026-10-20. This is a sufficient floating-point predicate for a particular cached plane
evaluation. It is not a new general dot-product theorem, an enclosure of original projected
geometry, or a proof about approximate interior alpha. Implementation:
`tgs_coarse_exact_window_v9.cuh`; frozen protocol: `TGS_WINDOW_CANDIDATES_PROTOCOL_V9.md`.

## Arithmetic and assumptions

For an edge, write c=(nx,ny,b) for the stored binary64 coefficients and h=RN32(c).
The local coordinates x,y are integers with |x|<=Bx<=8192, |y|<=By<=8192.
The integer pixel and integer anchor subtraction is exact: |anchor|<2^23,
W,H<=8192, and the runtime checks the cached local footprint.
Each nonzero component of both c and h has absolute value in [2^-60,2^20].
All values are finite. Invalid inputs take the original native64 path.

The specified evaluations use separately rounded operations:

    d32 = RN32(RN32(RN32(hx*x) + RN32(hy*y)) + hb)
    d64 = RN64(RN64(RN64(cx*x) + RN64(cy*y)) + cb).

The explicit CUDA rounding intrinsics specify this ordering even under the build's fast-math
flag. x,y conversion to binary64 is exact. The predicate concerns d64, not exact real geometry.

## Bound

Set B=(Bx,By,1), and define exact nonnegative real quantities

    D   = sum_j |c_j-h_j| B_j
    M   = sum_j |h_j| B_j
    M64 = sum_j |c_j| B_j.

For unit roundoff u32=2^-24 and u64=2^-53, the standard ordered dot-product bound gives
|d32-h dot (x,y,1)| <= gamma_3(u32) M and
|d64-c dot (x,y,1)| <= gamma_3(u64) M64, where gamma_3(u)=3u/(1-3u).
The three is the longest multiplication/addition rounding path for a term in this expression,
not the number of products plus additions. Triangle inequality and the coefficient residual give

    |d32-d64| <= D + gamma_3(u32) M + gamma_3(u64) M64
               <= D + 2^-22 M + 2^-51 M64 = E_edge.

The two relaxed constants strictly exceed their respective gamma_3 constants.
No power-function error estimate occurs here.

Under the stated coefficient domain, binary32 coefficients lie on the 2^-83 lattice.
Integer multiplication and cancellation of such terms cannot create nonzero values smaller
than 2^-83. Binary64 coefficients lie on the 2^-112 lattice. Thus the relevant ordered plane
operations have no nonzero subnormal result or overflow; the relative rounding bounds apply
despite the compilation environment's FTZ setting. A coefficient minus its nearest binary32
value is computed exactly in binary64 by Sterbenz's condition (or is zero).
Nonzero residuals are at least 2^-112; upward conversion to binary32 remains normal.
All positive bound operations, conversions and maxima round upward, so the stored bound
does not underestimate E_edge. The coefficient upper bound and coordinate bound keep
all these sums/products far below overflow. This domain is sufficient, not necessary.

Let E be the upward-rounded maximum of the three E_edge bounds. If max_k d32_k > E,
then its maximizing edge satisfies d64_k >= d32_k-E > 0. Therefore the native64
window is outside. No active-edge identity, tie convention or interior normalization is needed
to reject it. Otherwise the implementation evaluates the unchanged native64 window.

This proves only the conditional outside decision under these operation/domain assumptions.
Tile generation, projection, near-plane clipping, original geometry and the coefficients' own
construction are inherited from the reference and are not certified by this argument.

## Consequence for this implementation

The native reference discards a window with max_k d64_k>0 before alpha/compositing.
Replacing only a certified such discard cannot change a retained alpha, incoming transparency,
last/median contributor, cap/cutoff branch, maximum-blend contribution or forward accumulation.
Uncertified windows execute the original edge, normalization, power and cap arithmetic.
Induction in the unchanged per-ray candidate order therefore preserves its forward state,
assuming all unrelated operations, candidate lists and caches are identical.
This supports removing the earlier approximate-alpha preflight traversal entirely.
The predicate is shared with backward; certified outside windows contribute no native loss
gradient. Atomic accumulation order can still produce tiny VJP differences, so the numerical
VJP check is separate and does not require bitwise atomic-gradient reproducibility.

The exact rational cache audit checks all small fixtures and a prespecified real coefficient
subset; all visible real extents/domains are checked. This is implementation evidence for the
stated envelope, not exhaustive execution of every real pixel/edge combination. The seven
full-image forward comparisons separately test the realized paths and outputs.

## Limits on the research claim

This combines an established conservative predicate with exact fallback for the measured TGS
work distribution. The current work does not establish a novel general predicate, universal
speedup, geometric quality improvement, cross-scene performance or training improvement.
Independent precision and balanced operator-cost audits determine whether this implementation
is worth pursuing; they do not complete the broader scientific goal.
