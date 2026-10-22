# Coupled certificate transfers, but remains an analysis/reference tool

2026-10-21. Mathematical refinement and fixed-input CUDA diagnosis only. No training,
optimizer, backward, GT metric improvement, practical density rule or efficiency claim.
The full research objective remains open.

## Outcome

Three fixed variants were compared: original v17, signed omitted-tail difference,
and signed tails plus common rounded prefix and an exact-zero input-sequence case.
Formula/source hashes were frozen before v18 outputs. No outcome-driven threshold
selection or post-validation formula adjustment. All old/new error containments pass.

Development cohort: all64 old v17 pairs retained,13 nonzero,51 zero,20 early-stop pairs.
For nonzero changes, median bound/error decreases5.26810 ->1.42154 ->1.20173;
maximum decreases50.69837 ->5.26810 ->5.23091. All51 actual zeros are certified zero
by equal executed(alpha,RGB) inputs, rather than assigned a generic positive envelope.
This cohort was used in development, so it is not confirmation.

Separate event/camera cohort: flowers _DSC9136 and room DSCF4755, event3/4 per model;
64 paired pixels,12 nonzero,52 zero,13 pairs with native early stopping. These
cameras were previously used for SH evaluation, but not v14/v17 subdivision analysis.
The event indices exclude development0/1/2/5. Select up to8 geometric projected-parent
rectangle points and global SHA256 points to16/case, before rendering; no new image
error, alpha, visibility or bound selects pixels.

On these inputs, median nonzero bound/error1.72732 ->1.72732 ->1.38885, maximum
78.41354 ->33.92860 ->33.62117. All52 actual zeros receive exact-zero certificates.
The new cohort supports the conditional formula on other event/camera inputs in two
known models, not statistical independence, generalization across scenes, or a
whole-image safe splitting rule.

## Mathematical change and checks

Let H bound the full exact-composite change and L_a,L_b be the exact omitted tails.
Use H+|L_a-L_b| instead of H+L_a+L_b, then add the rounding envelopes. For a common
accepted prefix, the actual rounded color C0 cancels and its rounded T0 is shared.
With t0 the exact prefix transmittance and suffix composites S_a,S_b,

    (T0/t0)*(D_full-L_a+L_b) = T0*(S_a-S_b).

Only suffix arithmetic needs a new local rounding envelope gamma_(3l)(C0+T0*S).
The two valid bound families are intersected channelwise; identical whole accepted
input sequences imply exact native RGB equality under the fixed operation model.
Details and premises: TGS_COUPLED_CERTIFICATE_NOTE_V18.md.

An additional structural audit checks the exact rational prefix factorization for all
128 old/new pairs,768 channelwise local rounding bounds and103 literal zero pairs.
All pass; maximum actual local rounding residual/envelope is0.265294. This directly
checks each local envelope, not just the final summed containment. No compiler-general
proof is claimed; recorded update profiles and normal-range premises are inherited
from the independent endpoint audits on these fixed packets.

## CUDA evidence

Exactly6 native forwards (two originals and four held-parent-color splits),256 tiny
observer calls. All128 new RGB/alpha/last-ordinal endpoint replays literally match
native output. Baseline aux/radii/maxblend/contributors literally match retained v12
SH-only arrays. Precomputed-centroid RGB versus v12 native SH max differences are
2.38419e-7 flowers /1.78814e-7 room, below the unchanged2e-4 gate. Fresh split images
have no older sealed full-image reference, and no such comparison is claimed.
Models and camera tensors remain unchanged. Native and observer binaries are reused;
no new CUDA build, production edit or GPU rerun is needed in this phase. All executed
development, capture, fresh audit, structure and limitation-analysis jobs exit0.

## Why this is not yet a useful adaptive-splitting method

Worst new ratio: room event4 pixel(1404,811), native max-channel change4.78029e-5,
coupled bound0.00160719, ratio33.62117,78 crossed external layers. The signed window
term is approximately[-3.873e-4,-6.252e-4,-6.392e-4], while the signed order term is
[4.350e-4,6.701e-4,6.849e-4]. Their cancellation leaves about[4.773e-5,4.491e-5,
4.575e-5] before native rounding. Separate absolute-value bounds lose this cancellation.
The observed limitation is not evidence that the exact signed decomposition fails.

Development accesses59,437 candidate records /9,687 accepted /409 divergent suffix
records. New cohort accesses61,022 /9,724 /644 respectively. The short divergent
suffix is a structural opportunity, not a measured reduction in total work. Current
code still needs counterfactual child alphas/order, full candidate records, rational
composites and tails; the six full trial forwards are part of data acquisition.
No runtime or memory saving has been demonstrated. A tighter bound alone is not an
efficiency or quality result, and sampled-pixel containment cannot certify a whole split.

Next implementation should test an incremental counterfactual query: reuse original
scene preprocessing/tile lists, preprocess only four children, merge their depth keys
with unchanged external records in the affected region, and evaluate the signed RGB
change with native stopping. Validate against full native split renders and then freeze
a balanced timing protocol. This would test an actual local-computation speed hypothesis;
if it offers no useful advantage, reject it rather than extending the bound indefinitely.
No benefit, novelty, optimized implementation or timing is established here.

## Prior work and reproducibility

Fixed-slot analytic rendering-error quantification and local cumulative evaluation
already appear in [GaussianPOP](https://arxiv.org/html/2602.06830).
[PLEDGE](https://arxiv.org/html/2610.17605) explicitly distinguishes parameter/footprint
bounds from pixel-value guarantees and conditions its composited-color bound on fixed
geometry/alphas. These are related limits, not evidence for novelty of our identities.
Search coverage is targeted, not exhaustive. Automated hypothesis pages and secondary
paper critiques were not used to establish claims.

Outputs: experiments/tgs_coupled_v18, including all raw new packets, frozen plans,
three-variant results, structure_audit.json and limitations_audit.json. Core source:
tgs_coupled_certificate_v18.py; capture: trace_tgs_coupled_fresh_v18.py; final audits:
audit_tgs_coupled_fresh_v18.py / audit_tgs_coupled_structure_v18.py. The open
median_depth_note.tex is untouched. Provenance is sealed by seal_tgs_coupled_v18.py.
