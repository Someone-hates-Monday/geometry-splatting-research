# Power window conditioning and exact tangential counterexample

This note records standard elementary mathematics applied to the pinned TGS
window. It does not assert that these inequalities or mixed precision are novel.
All image discrepancies below compare implementations of a fixed model, not GT.

Let u=max(0,-phi/r), r>0, and a(u)=min(.99,o*u^s), o>0, s>0.
The following bounds precede discrete contribution cutoff and compositing branches.

For 0<s<=1 and nonnegative u,v,

    |a(u)-a(v)| <= o*|u-v|^s.

Proof: assume u=v+d, d>=0. For v>0 the derivative of
(v+d)^s-v^s is s*((v+d)^(s-1)-v^(s-1))<=0. Its maximum is
its limit at v=0, d^s. Exchange u,v for the other case. Multiplication by o
and the 1-Lipschitz cap preserve the inequality. At s<1, the uncapped derivative
o*s*u^(s-1) diverges at zero; no finite global Lipschitz constant exists on an
interval containing zero. Continuity alone therefore gives a weak uniform bound.

For the stored s=.019999999552965164, o=.9998999834060669, this global
Hölder sufficient budget for alpha error<=3e-5 is

    delta_u <= (3e-5/o)^(1/s), log10(delta_u) = -226.1417703773… .

This is a worst-case sufficient bound, not a necessary precision requirement
for an individual pixel. It does not imply that all pixels need 227 digits.
For a certified interval 0<l<=u,v<=h the mean value theorem instead gives

    |a(u)-a(v)| <= o*s*l^(s-1)*|u-v|  (0<s<=1),
    |a(u)-a(v)| <= o*s*h^(s-1)*|u-v|  (s>=1).

At the exact boundary fixture u=377/8000000000, the local derivative is about
3.03e5: demanding alpha error3e-5 needs u precision of order1e-10 locally,
which is practical with adequate higher precision. The global bound is highly
conservative because it also covers arbitrarily small positive u.

The exact counterexample uses the original stored world points
(-1.4921875,-1.4921875,2), (1.0078125,-1.4921875,2),
(-1.4921875,.3828125,2), image256x256 and identity pinhole numerator matrices.
With EPS=1/10^7, the projected right triangle is the homothety of
(32,32),(192,32),(32,152) toward (127.5,127.5), factor t=2/(2+EPS).
Its hypotenuse satisfies 3x+4y=704+188.5*(1-t).
Pixels (176,44),(188,35) have 3x+4y=704, so both have

    phi=-377/200000010 < 0, r=40*t,
    u=377/8000000000,
    alpha=.713545368760090565906… .

The exact stored float32 normal/offset at both pixels gives phi=-45/16777216;
the pixel displacement (12,-9) is exactly orthogonal to the stored normal.
All their tiles are present, yet CUDA emits alpha0 and .7236766815. Native
inclusion/cutoff/early-stop cannot explain this by the same exact edge value:
in a one-triangle scene the ideal alpha is above cutoff, below cap, and far
below the early-stop opacity. Arithmetic in evaluating the edge is material.
Vertex quantization alone is not a sufficient causal explanation, since the
stored-vertex polygon still contains both pixels.

An adaptive candidate should propagate intervals for u and alpha, use the local
bound above to select arithmetic precision, and separately certify inside/cap/
alpha-cutoff/early-stop decisions. If an interval straddles a discontinuity,
the usual smooth bound cannot be applied across that branch. The contribution
cutoff itself has a jump of approximately1/255 and can exceed a3e-5 tolerance
even with s>=1. Geometry preprocessing cutoffs are separate discontinuities.

Possible implementation steps, not yet validated: evaluate cached float
coefficients in double to isolate pixel arithmetic; construct high/low or double
geometry coefficients from original projection to isolate coefficient error;
then use a certified fallback near uncertain edges instead of making every
pixel pay the same cost. Backward must differentiate the resulting mathematical
window consistently. A floor/clamp on s changes the model and requires a new
scientific hypothesis and training comparison, not a numerical-correctness label.

Machine-readable evidence: experiments/tgs_stable_window_v2/
small_sigma_boundary_certificate.json, small_sigma_native_cache.npz,
small_sigma_independent_audit.json, full_old_image_diagnosis.json.
