# v11 fixed-input GT identity and zero-step image compatibility

2026-10-20. Scope remains math and small CUDA diagnostics, no optimizer/full training/tuning.
Preserve all v9/v10 evidence read-only. This phase asks whether the operator change has actual
image-value sufficient to motivate its remaining cost, not whether matching an oracle is a paper.

## Cohort fixed before GT inspection

Exactly the two test cameras already selected before v10 output: flowers/_DSC9128 (test index11)
and room/DSCF4915 (test index31). Keep the same iteration7000 checkpoints, resolution/FoV,
SH3/effective inference mask1/background, and zero optimizer steps. No camera/checkpoint or
metric substitution after GT/quality outputs. These are two convenience diagnostic views, not
a complete test-set benchmark or independent training seeds.

Official source: https://jonbarron.info/mipnerf360/ links Dataset Pt.1/Pt.2; extract only the
selected images_4 flowers and images_2 room files. Fetch the corresponding published TGS
test/ours_30000/gt/00011.png and00031.png as predicted by the pinned loader/render order.
GT is an original photograph, independent of the rendering checkpoint's7000/30000 step.
The integer index alone does NOT prove that identity.

Before image payloads, archive the current official project HTML, exact dataset URLs/HEAD
identity and ZIP directory metadata; freeze chosen members, CRC/size and source hashes.
Use only public HTTP Range, no credentials/whole-archive fallback. Per request<=16MiB,
combined successful new cached Range payloads<=96MiB; directory-only phase<=8MiB new data.
Verify status206, Content-Range, object size/ETag, ZIP CRC and SHA256. Record actual failures,
resume only on verified terminal failure without overwriting completed files. Do not upload data.

## Identity prerequisite

Decode source photograph and published GT with the recorded Pillow environment into RGB.
No resize/crop/registration/color/exposure transform or EXIF transpose is allowed to manufacture
a match. Require the same dimensions and exact decoded RGB bytes. Check pinned source
canonical split/order and filename-to-index claim separately. If any view fails, retain it as an
identity failure and do not run a misleading quality comparison for the unresolved cohort.
Decoder/pyramid differences, if observed, need a distinct documented diagnostic rather than an
after-the-fact tolerance increase or nearest-looking view substitution.

## Fixed quality comparisons after identity passes

Compare readonly v9 coarse and native64 oracle and the historical piecewise module SHA
f06b1a1ccf5c11f99cebda0b0b5338ca273186ba316aa74fa619c4fa45ccd852, same model/input.
The latter has different numerical/geometry code; this is operator compatibility on frozen
public weights, not a controlled training-method ablation or proof of a precision-only effect.
Recapture oracle/coarse against retained v10 forward arrays bytewise before scoring. Preserve
all raw RGB arrays and pointwise differences, input hashes before/after, module/source hashes.
No stochastic optimization, novel-view camera adjustment or timing rerun in this phase.

Primary output follows the published PNG saving convention: float32 raw RGB*255,+0.5,
clamp[0,255],uint8 (separate roundings, same operation order as torchvision.save_image).
Compare uint8/255 against verified uint8 GT/255. Record also secondary clipped float RGB
quality to expose quantization effects, plus raw min/max/nonfinite/clipping fractions.
No selective crop, masking, postprocessing or choosing the better quantization definition.

MSE is full RGB mean squared error in binary64, data range1. PSNR=-10log10(MSE), with exact
zero reported as infinity. Report each view separately, MSE and PSNR differences, and their
two-view arithmetic mean only as a descriptive convenience, never a dataset score.
SSIM, if implemented, uses a fixed11x11 normalized Gaussian sigma1.5, channelwise float64
means/variances, C1=.01^2,C2=.03^2, same-size zero padding5, averaged over all RGB pixels;
report this boundary convention explicitly. Freeze implementation/hash before scoring.
No LPIPS score or metric substitution without a separate explicit version/provenance plan.

Independently reconstruct PNG conversion and recompute MSE/PSNR and selected SSIM statistics
from retained arrays before conclusions. Coarse/native64 quality equality follows from their
bytewise forward equivalence and is not a new GT improvement. Compare piecewise fairly with
the same image/quantization convention and retain positive/negative/near-zero changes.

## Interpretation and next decision

Do not set a post-outcome significance threshold or claim universal improvement from two views.
If quality changes are negligible or adverse while cost is several times the historical operator,
the evidence favors keeping native64/coarse as a diagnostic reference and prioritizing another
geometry/optimization mechanism. That is a research-direction judgment, not a causal estimate
of retrained quality. Public weights may compensate for their original renderer's arithmetic;
zero-step incompatibility cannot by itself rule out a correctly retrained method.
Full novelty, geometry, broader scenes/views and training validation remain unmet requirements.
