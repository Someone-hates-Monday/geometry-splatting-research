# v28 r1: preserve the numerical appearance of unedited primitives

Written after the original calibration full-reference gate failed, before any heldout
payload was fetched. The original protocol, executable, choices and failure remain
unchanged. This is an explicit numerical-contract correction, not a new success of
the failed original gate. No scores, candidates, selection rules or thresholds change.

The first10 flowers native references passed. Room calibration_0 parent42674 failed
on the11th full-reference call. A separately frozen four-forward diagnostic reproduced
the baseline and cached candidate exactly, then found all9 visible external geometry
fields and all9 visible child geometry fields literal. In contrast, reevaluating
Python SH colors for the whole N+3 model altered790593 color values, peak4.76837e-7;
the RGB mismatch affected1021757 pixels,1020501 outside the candidate rectangle.
Thus it changes unedited numerical appearance and violates the cached oracle's fixed
external-color contract. Holding original FP32 colors fixed and evaluating only the
four children produced a complete native render literally equal to the cached image.
The lower-level PyTorch operator causing the shape-dependent roundoff is not yet
isolated; do not attribute it definitively to mean, norm or SH polynomial operations.

For the corrected full reference, evaluate original colors once for each unchanged
camera/model, retain every unedited original FP32 color in relative order, and append
the same four child-only SH colors used by the candidate bank. SH coefficients and
camera directions retain the coherent mathematical convention. This explicitly
defines the finite-precision counterfactual; it does NOT certify a pipeline that
reevaluates every original color after changing array shape, historical native SH,
training, gradients, or generalized appearance-update equivalence.

Reuse the10 already-passed flowers reference patches after independent reconstruction
and hash checks; do not rerender them. Reuse the room42674 calibration_0 full inherited-
color render saved by the four-call diagnostic, verify its full RGB hash and outside-
ROI equality before saving a reference patch. Perform the remaining7 calibration
references and at most18 heldout references using the corrected contract. Preserve
the failed original call and all4 diagnostic calls in counts:8 calibration scoring+
11 original reference attempt+4 diagnostic+7 remaining calibration references+
8 heldout scoring+18 heldout references=56 total native calls if all gates pass.
The old reference attempt exited1; the corrected receipt explicitly records recovery.

Stop on any further mismatch; do not loosen RGB or scalar gates. Test image selection,
candidate selection and policies remain frozen. The transfer result, if completed,
must identify this contract and failure history, rather than silently presenting it
as a validated whole-model color recomputation experiment.
