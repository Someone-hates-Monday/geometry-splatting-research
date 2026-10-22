# Restricted SH-center semantics: coefficient-only conversion and chain-rule limits

2026-10-21. These are elementary real-arithmetic consequences of the pinned source expression,
not claims of a new general theorem, image-quality bound, or complete native-gradient analysis.
Exact substitutions/radicals are checked by verify_tgs_sh_semantics_math_v12.py (SymPy1.14).

## Premise from source

The pinned forward source passes `(glm::vec3)(p.x,p.y,p.z)`. The parenthesized comma expression
evaluates to the last scalar p.z; pinned GLM's scalar vec3 constructor fills all three components.
Thus the legacy SH evaluation center is q(p)=(p_z,p_z,p_z), while the corrected constructor
passes p=(p_x,p_y,p_z). The archived source and GLM file hashes are recorded in math_results.json.
This source premise is distinct from proving that a particular historical training run used it.

For camera c and nonzero distances, write d=(p-c)/||p-c|| and legacy direction e=(q-c)/||q-c||.
With delta=q-p and r=||p-c||, e=normalize(delta+r*d). A replacement that retains only d loses
the r information in this expression unless special conditions make it irrelevant.

## Explicit coefficient-only obstruction

Choose p=(1,0,0), so q=(0,0,0), and cameras c1=(1,0,-1), c2=(1,0,-2).
Both corrected directions equal(0,0,1). Their legacy directions are(-1,0,1)/sqrt2 and
(-1,0,2)/sqrt5. The admissible degree1 scalar color field L(e)=1/2-e_x/4 gives

    L1 = 1/2 + sqrt2/8,
    L2 = 1/2 + sqrt5/20.

Both lie strictly inside(0,1), so the counterexample does not rely on color clamping. Degree1
is contained in the actual degree3 SH family; its coefficient can absorb the positive basis
normalization constant. Any single new direction-only function H(d), including any finite-degree
SH expansion with camera-independent converted coefficients, takes ONE value at the common d.
It cannot equal both L1 and L2. This disproves a universal coefficient-only conversion for all
admissible legacy fields/cameras; it does not say every particular field is impossible to convert.

For these two samples, triangle inequality implies

    max(|H(d)-L1|,|H(d)-L2|) >= |L1-L2|/2
      = sqrt2/16 - sqrt5/40 = 0.032486648210823698... .

The midpoint H(d)=(L1+L2)/2 attains this bound, and is representable by a constant SH field.
The bound is for this constructed single-channel color-field example. It is NOT a PSNR floor,
an empirical bound on flowers/room, an opacity/compositing theorem or proof of a quality cause.
Constant fields, zero center displacement and restricted camera sets can avoid the obstruction.

## Translation and a conditional gradient-field example

For common translation t=(tx,ty,tz), the corrected p-c is unchanged. The legacy q(p+t)-(c+t)
changes by(tz-tx,tz-ty,0). Hence generic common translations do not preserve the legacy
direction; diagonal translations or special colors can still be invariant. Earlier runtime
translation probes remain separate empirical evidence, not new observations in this proof.

The center map has Jq=[[0,0,1],[0,0,1],[0,0,1]]. For an isolated scalar color F(q,c), the true
centroid derivative is Jq^T grad_q F, not grad_q F. To state a concrete conditional limitation,
assume a reported isolated color gradient uses the latter identity-chain form. Let

    F(q,c)=1/2-(q_x-c_x)/(4||q-c||), c=(1,0,-1).

At p=(1,0,0), q=0, that assumed reported vector is(-sqrt2/16,0,-sqrt2/16), while the true
chain-correct vector is(0,0,-sqrt2/8). The reported vector field composed with q(p) additionally
has dG_x/dp_z-dG_z/dp_x=-sqrt2/8 at this point. On a smooth neighborhood, this isolated assumed
field therefore cannot be the gradient of any C2 scalar potential.

This checks the consequence of an explicitly stated gradient form. It does NOT by itself
identify every native backward contribution, prove the complete native force field has that
curl, or establish training failure. A runtime isolation/finite-difference test is required for
such code-specific claims; geometry/opacity terms may add other derivatives or errors.

## Research use

Treat a forward-model correction and a legacy-weight-compatible conversion as different tasks.
The source-only four-view experiment tests which edit explains the measured compatibility
change. This note constrains one possible coefficient-only remedy. A useful new method still
requires related-work review and evaluation beyond a code fix or generic information-loss proof.
