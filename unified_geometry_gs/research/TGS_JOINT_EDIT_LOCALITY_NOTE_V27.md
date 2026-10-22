# Conditional additivity of fixed topology edits

2026-10-21. A source-conditioned finite-dependency argument,not a new general theorem
or a formal proof of all CUDA compilation behavior. No training/generalization claim.

Fix one original scene,camera,background,GT and rendering convention. An edit e removes
its own parent and appends its four fixed children with fixed attributes/colors. For
each e let R_e enclose the native tiles of that parent and its children. This is a
STRUCTURAL support superset;an observed zero RGB difference does not make R_e empty.
An invisible parent and invisible children can have structurally empty R_e.

Assumptions:primitive preprocessing depends only on its own inputs/camera;unmodified
primitive attributes stay fixed;original records keep their relative order;ties between
old external records and appended children follow the same stable rule;each edit has
a distinct parent;only its own children are inserted. Window,alpha cap/cutoff and
skip-current early stop are unchanged. These conditions are checked from frozen source,
visible cache fields and actual image comparisons,not assumed from geometric area alone.

At a pixel outside every R_e,all scene records are unchanged up to ID remapping. At a
pixel in exactly one R_e,the joint edited list is the same as for e alone:all other
edited parents/children are absent from that tile. Thus if the R_e are pairwise disjoint,
the joint RGB is a mosaic of the single-edit RGB arrays and the original array. Even
the native stop decision is identical because the entire local sequence is identical.
Algebra is not by itself a guarantee of FP32 bits;the real-image study checks the mosaic
literally and checks unchanged/child cache fields under actual joint ID remapping.

For any pointwise loss phi and deterministic pointwise quantizer q,including raw L1,
MSE and PNG8 SSE,each pixel then has at most one nonzero loss-change term. Consequently

    L(X_S,G)-L(B,G) = sum_{e in S}[L(X_e,G)-L(B,G)].

For SSIM radius r=5 define A_e=(R_e dilated by r) intersect image,with empty R mapped
to empty A. At any SSIM output outside A_e,its neighborhood contains no edited input.
If A_e are pairwise disjoint,each SSIM output can depend on at most one edit,so the
same additivity holds for mean SSIM and the L1/L2+DSSIM forms. Disjoint R alone does
not suffice for SSIM if the halos overlap. The condition is sufficient,not necessary:
overlapping supersets may still give zero interaction because of occlusion/cancellation.

For multiple fixed cameras,use an edge between candidates if their structural loss
supports overlap in ANY scored camera (or if they edit the same parent). A set with
no edges satisfies the per-camera conditions,so any fixed weighted sum over those
cameras is additive in exact arithmetic. This guarantees neither an unseen-camera
improvement nor usefulness after parameter optimization,when supports/attributes change.
No independent-set selection or model acceptance is performed in v27.

For a pair define interaction eta=Delta_pair-Delta_i-Delta_j. Pointwise-support disjoint
pairs must have exact integer PNG8 SSE additivity and literal RGB mosaic equality.
SSIM-support disjoint pairs additionally must have literal GPU SSIM mosaic equality.
For reductions of already computed equal FP64 pixel/map terms,the conservative bound

    |eta| <= (gamma_n+8*eps)*(sum|t_pair|+sum|t_i|+sum|t_j|)/n,
    eps=2^-52, gamma_n=n*eps/(1-n*eps), n=3HW,

covers each fixed sum/division and final contrast rounding. eps conservatively exceeds
unit roundoff. Exact per-element identities are a separate prerequisite;the bound
does not certify ideal real SSIM or excuse wrong images/maps. A deterministic CPU
full-image reference is also checked with unchanged v24 tolerances. Interacting pairs
have no zero-interaction assertion;all signed values are retained.

Why observed RGB supports are unsafe:consider an ideal exact-arithmetic scalar blending
sequence with skip-current threshold1/10000. Two front edited layers have alpha49/50
and color1;the remaining third has alpha99/100 and color98/99. Original RGB is2499/2500.
Deleting either front layer alone gives the SAME RGB,while deleting both gives49/50.
This exact rational generic deletion example has empty observed single-change supports
but a nonzero joint effect. It is a software counterexample,not a real triangle-split
result;no FP32 implementation identity is asserted for the rational colors.

The study freezes all six pairs of four known moved-color parents in each of four
known views,including empty and unfavorable cases. H/M alternatives of the same parent
are not simultaneous edits. No broad novelty or density-policy efficacy follows from
this limited proof and fixed-state diagnostic.
