# Finite-support image-loss differences for fixed scene edits

2026-10-21. Data-informed mathematical/CPU addendum after the v23 second-view and
float-loss results; not part of the earlier prospective diagnostic. No new render,
training or assertion of novelty. All32 known case/view combinations are retained.

Let B and X be full RGB arrays on the same H-by-W grid, and assume their difference
is supported in the known rectangular ROI R (a superset of changed pixels is allowed).
GT G and all boundary conventions stay fixed. This assumption follows from the
source-conditioned local replay/list argument and is checked by full-image hashes
and pixel equality; it is not an arbitrary bound on geometric displacement.

For pointwise loss l, exactly in real arithmetic,

    Delta L = (1/(3HW)) sum_{u in R,c} [l(X_uc,G_uc)-l(B_uc,G_uc)].

This covers raw L1, squared RGB error and the explicit PNG8 quantizer, with no Taylor
approximation. Signed PNG8 squared-error differences are exact integer sums under
v22 bounds. Floating point reductions may round differently when zero terms/order
change; use an explicit summation error bound, not an integer-exactness claim.

For the v11 SSIM map each channel uses a fixed separable11-by11 Gaussian (radius r=5),
same-size zero padding and local moments mu_X,mu_G,E[X^2],E[G^2],E[XG]. Each map value
S(X,G)_u depends only on X/G within u+[-r,r]^2. If u is outside A=R dilated by r and
clipped to the image, that neighborhood contains no changed input, so

    S(X,G)_u = S(B,G)_u.

Therefore Delta mean SSIM equals sum_{u in A,c}(S(X,G)-S(B,G))/(3HW). To compute all
map values in A, crop both images and GT to J=A dilated by r and clipped to the image,
i.e. at most R dilated by2r. Evaluate the SAME SSIM definition on J, then select the
central A outputs. No selected output sees an artificial crop boundary: every needed
sample is in J, or is outside the original image and has the same zero padding. A
single5-pixel input halo is insufficient when outputs up to5 pixels outside R are
included; the sufficient input halo is10 pixels around R.

The raw image objective forms .8*L1+.2*(1-SSIM) and .8*MSE+.2*(1-SSIM) then have local
differences .8*Delta(pixel loss)-.2*Delta(SSIM), with the SAME whole-image denominator.
This says nothing about global/per-scene regularizers, losses with larger/global
support, geometric normal reconstruction, backward gradients or stochastic crops.
It does not establish the actual historical public checkpoint training branch.

CPU implementation check: reconstruct all32 actual RGB arrays from their lossless
patches and validate full reference hashes; assert outside-R equality; assert full
SSIM differences outside A are literal zero and local/full SSIM map values in A are
literal equal. Compare three local reductions to the pre-existing full-array float64
results using an explicit worst-case rounding bound. For n full terms and m retained
terms, epsilon=2^-52, gamma_k=k*epsilon/(1-k*epsilon), bound

    (gamma_n+gamma_m+4*epsilon)*sum(abs(retained terms))/n.

This uses a conservative floating addition bound (NumPy may sum pairwise); the4epsilon
allowance covers division/endpoint rounding. All per-element loss/map terms must be
literal equal first. No tolerance can rescue a changed map, wrong halo or wrong hash.
For zero terms the difference must be literal zero. The bound concerns reductions of
computed FP64 terms, not their distance to ideal real SSIM or GPU FP32 arithmetic.

This is a standard finite-dependency argument supporting a loss-aligned implementation,
not a new SSIM theorem or evidence of model-quality gain. The CPU check is not a speed
benchmark; the kernel replay needed for X in R and reusable baseline state still cost
work. A future CUDA implementation must define precision, parent removal, stable order,
finite halo and complete setup/memory costs before claiming useful speed or quality.
