# v23 fixed-proposal second-view and loss-alignment diagnostic

2026-10-21. Use installed hypothesis-generation and scientific-critical-thinking guidance.
Local public research inputs only. Human project owner authorized mathematical checks
and small CUDA diagnostics; no training, backward, accepted edit, cloud or new download.

Observation already seen: all16 fresh-view v20/v22 midpoint splits and PNG8 SSE scores,
3 negative13 positive. Their correctness and warm-cache timing are known. Old test
cameras/GT/baseline images were already explored in earlier phases. The effects of
these specific v20-selected parents on old test views are not yet measured here.
This is a prospective local diagnostic of these new contrasts, NOT a registered
study or untouched independent-scene generalization test.

Question: do the same geometric edits have consistent benefit/rank in a second known
camera, and does PNG8 SSE have the same sign/rank as float image objectives?
Candidates: view-stable geometric benefit; visibility/order dependence; color-direction
dependence; or quantization/objective mismatch. None is assumed established. No attempt
will automatically accept/reject a scientific hypothesis or a scene edit.

Freeze all four parent IDs per scene selected in v20, both hold/moved appearances and
the v11 old_0 camera, without inspecting new outcomes. Four moved cases per scene are
primary: the same inherited child SH coefficients are reevaluated at each camera.
Hold is an attribution diagnostic using camera-specific parent color, NOT one coherent
camera-independent SH parameterization. Keep all8 cases/scene, including invisible edits
and zero effects. No candidate replacements, new views, extra seeds or performance timing.

Reuse unchanged v21 control and v22 scalar-score binary. At old_0 perform exactly two
baseline + two32-child bank + sixteen full-reference forwards =20 total. Gate original
baseline against v14 precomputed-color image, old/fresh GT JPEG/PNG/NPY identity, bank
geometry/list and debug whole RGB/alpha/path literal equality, GPU scalar/full and CPU
PNG8 score equality. Save raw lossless RGB ROI patches, hashes, camera identity and
bank parameters. Mixed bank-render RGB is discarded. No recompile or production change.

Use CPU reconstruction of old test patches and saved v20 fresh patches for four-view
metrics; verify full-image hashes first. Primary outcome: signed PNG8 SSE change and
sign consistency for4 moved proposals per scene. Secondary: raw (unclipped) RGB float64
MSE, L1, SSIM, and diagnostic blends .8*L1+.2*(1-SSIM), .8*MSE+.2*(1-SSIM).
SSIM uses v11 Gaussian11 sigma1.5, full RGB mean, zero padding. Compute delta by summing
per-element differences, avoiding cancellation from subtracting rounded global means.
These float64 re-evaluations match image-objective forms, not GPU FP32 training arithmetic
or the full objective containing geometry/opacity regularizers. The archived TGS code
chooses L2 for large_scene and outdoor, otherwise L1. Saved public cfg does not prove the
actual historical branch or every training hyperparameter. Do not assert it does.

Describe all signs, numerical magnitudes, paired ranks and six pairwise ordering relations
per four-proposal group. Ties stay ties, do not jitter. No p-values, pixel-as-replicate
tests or population confidence intervals. Four proposals/scene and two dependent cameras
are too narrow for a broad policy claim. No numeric sign threshold/exclusion chosen after
seeing results. Report exact numerical zeros and near-zero values without implying
practical importance. Cross-view inconsistency challenges a universal transfer claim;
consistent finite examples do not prove it. Quantization and float-objective discrepancies
are descriptive, not proof of a unique mechanism.

Freeze runner, analysis, protocol, models/cameras, GT, old/fresh raw sources and build
receipts before new rendering or new float loss analysis. Hash all. Stop after the20
new forwards and CPU audit; retain any failure and do not rerun a job on timeout alone.
