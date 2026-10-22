# Candidate extension through loss of a finite median

2026-10-15, before extension test results. This follows the one missing
root in each of shared_v1 full/geometry step.01; those original branch
means remain undefined, not retroactively assigned a value.

Let Omega=sum[-log(1-alpha)] on 0<alpha<1,rho>0, fixed finiteN.
Use the original median loss for Omega>log2. For Omega<=log2, define
Lext=(log4-Omega)^2. Claim: this defines a C1 function on the entire
open coefficient domain and agrees exactly with every finite-root loss.
At any Omega=log2 point, H(maxp)<Omega=log2 strictly. By continuity this
inequality persists in a neighborhood, and whenever a median exists
there it is beyond all peaks. Its original loss is exactly the same
(log4-Omega)^2, establishing an open-neighborhood formula across the
boundary. This is a proposed mathematical extension, not a constant
missing-ray reward and not a GT/convergence/novelty claim.

Checks: 16 fixed random models seed13017 N1/2/4/8, desired Omega spanning
.3/.65/.7/1.0/1.5/3.0; MP240 exact coefficient gradients versus independent
root-composed central differences1e-30/1e-35, relative<=1e-30. Single-layer
alpha=.5 exact boundary and two-layer alpha1=.2, alpha2=.375+/-1e-k for
k2/8/16/32, separated peaks, must agree with the tail-region closed form
and gradient on both sides to1e-120. Preserve the separated thin-gap
vanishing-gradient counterexample: finite-root behavior is unchanged.

Revisit exactly the two missing-root updates from shared_v1 full/geometry
step.01 using the saved shared direction and IDs. Recompute true covariance
inverse coefficients in MP240 from the continuous raw extension, classify
Omega and compare Float64 root-absent formula/its full11 raw autograd
directional derivative with re-evaluated MP h1e-8/1e-10 differences.
No source models change. This audit validates definition/derivative on
those updates; it is not the entire shared branch re-evaluated in MP.
Report any classification disagreement rather than treating a numeric
unbracketed status as mathematical absence of a root.
