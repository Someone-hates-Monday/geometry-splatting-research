# v12 source-isolated SH compatibility and exact semantic counterexamples

2026-10-21. Mathematics and small fixed-input CUDA only; no optimizer, retraining, clock
benchmark or method-level quality claim. v11 is sealed. Its two old views are explicitly
post-hoc explanation inputs, not fresh validation. One additional test view per flowers/room
is selected BEFORE outputs by minimum SHA256('tgs_sh_isolation_v12/'+scene+'/'+name), excluding
all previously selected v10 camera names. Keep every chosen view and report old/fresh separately.

## Source intervention

Primary pair: unedited archived_native9d974f09... and existing SH-only4fa625a4... . Assert that
its generated forward/backward sources are byte/text-identical to the pinned originals except
the ONE constructor expression replacement in each file:
`(glm::vec3)(center_triangle.x, center_triangle.y, center_triangle.z)` ->
`glm::vec3(center_triangle.x, center_triangle.y, center_triangle.z)`.
Verify build receipts, flags, native/GLM commits and SO hashes; never rebuild old sources.
Other forward comparisons: existing geometry-fixed955dd08a..., piecewisef06b1a1c..., coarse81d18c3c... .
These are diagnostic steps with additional changes, not isolated precision-only treatments.

Same public7000 weights/activations/background/SH3, same cameras and source resolution; no
parameter or image adjustments. Old original/coarse/piecewise views must match retained v11
outputs literally. Save all RGB/7aux/radii/maxblend/tile counts and defined last/median paths.
For original vs SH-only expect all non-color defined outputs to match bytewise; record whether
RGB changes. Do not force an effect if a view has no appreciable direction-dependent signal.

Two negative controls on the primary pair, on all four inputs: degree0 with the same DC weights;
and SH bypassed by constant precomputed RGB(1/4,1/2,3/4) for all triangles. Full forward output
should match bytewise in both controls. These images are not scored as competing quality models.
If a non-color/negative-control mismatch occurs, retain it and withhold the isolated attribution;
do not change threshold or camera. At most20 main and16 control forwards, no backward or timing.

## Fresh GT and metrics

Reuse already verified v11 photographs for old views. For each fresh view, freeze exact source
JPEG/GT members before payloads, HTTP Range<=96MiB total successful new payloads, CRC/SHA,
same no-whole-archive/immutable-identity rules. Require exact decoded RGB identity with no
resize/crop/registration/exposure/color transforms, plus canonical filename/test-index/source
order. A failure blocks quality scoring for the full cohort; it is not permission to swap images.
Use unchanged v11 PNG-primary/clipped-float-secondary MSE/PSNR/SSIM implementation and
independent integer/Torch/direct-patch checks. Freeze all new script hashes before renders/scores.
Report every view separately; no dataset/seed p-values or training-effect inference.

## Mathematical work and claim boundary

Pinned GLM scalar constructor maps a scalar s to(s,s,s), so the comma-expression legacy center
is q(p)=(p_z,p_z,p_z), not p. Verify this source premise independently of quality outcomes.
Construct an exact real-arithmetic counterexample where two camera positions give the same
corrected direction but distinct legacy degree1 SH colors. This can rule out a universal
coefficient-only conversion for that admissible field, not all fields or a measured PSNR floor.
Check the chain rule J_q^T on an isolated SH-color path, and any claimed nonconservative field
only under its explicit assumed reported-gradient form. Do not extrapolate to complete native
gradients/training dynamics without a distinct runtime test.
Use SymPy exact rationals/radicals, explicit nonzero-distance assumptions and source hashes;
do not present a generic chain-rule/information-loss argument as a novel general theorem.

## Interpretation

The primary source-only edit plus negative controls and fresh views can support a software
mechanism for observed zero-step compatibility changes on these inputs. They cannot prove
that published weights were trained by this exact commit or that correcting the model and
retraining helps/hurts. A well-supported direction must still connect to a useful method or
mathematical contribution and broader evidence; the overarching goal remains incomplete.
