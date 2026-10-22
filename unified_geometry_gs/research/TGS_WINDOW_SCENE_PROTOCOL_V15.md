# v15: isolated window response versus full scene, fixed existing cases

2026-10-21. This follows observed v14 outcomes and is a mechanism study on the SAME cases,
not a fresh holdout or a new scene replication. All32 v14 event-camera cases retained,
including13 zero cases. Use exact saved parent/four-child points, sigma, opacity, original
camera view/projection/center and resolution. Current authorization: math/small fixed-input
CUDA, no training/optimizer/benchmark/download/build.

Question: how much isolated alpha/window redistribution exists for the same splits, and
can full-scene G be explained by a simple fixed-slot alpha replacement model? Distinguish
screen-window response from actual GT quality. Published weights may require adaptation
after any kernel change; no quality implication is assumed.

64 new forwards maximum: parent alone and four children alone on each of32 cases, using
existing unedited native SO9d974, black background and precomputed white RGB. No SH.
All children copy sigma/opacity; no additional kernel alteration. Save lossless crop with
full-image reconstruction hashes for alpha and white RGB, radii and tile counts. Check:
white RGB channels literal same; max|whiteRGB-alpha|<=2e-6; parent/children preprocess radii
literal equal to v14 values. Gate failure preserves evidence and withholds effect claims.

Reconstruct full isolated alpha P and Q. Primary outcomes: RMS/MAE/max of Q-P, number of
changed pixels, sign distribution and sum(P),sum(Q). Pixel sums are discrete finite-view
quantities; do NOT equate them with the already-proved continuous mass invariance. Record
full-scene RMS(G), isolated alpha RMS, their descriptive ratio when denominator>1e-14.
This ratio includes color contrast, transmittance and all scene effects, not 'occlusion fraction'.

Conditional real-arithmetic identity: if the event occupies one contiguous slot with the
same prefix/suffix before and after, constant color c, prefix transmittance T and suffix
color B, then DeltaRGB=T*(c-B)*(Q-P). Since native precomputed colors are nonnegative,
abs(DeltaRGB_channel)<=Cmax_channel*abs(Q-P), where Cmax is max of all baseline colors
and black background for that camera. As an EMPIRICAL diagnostic only, count pixels where
the observed v14 |G_channel| exceeds Cmax*|Q-P|+3e-5. This is not an FP32 interval
certificate. Violations challenge the simple model (ordering/stop/numeric differences are
possible); passing does not prove order unchanged or isolate window causation.

No outcome improvement gate; keep negative and cancelling cases. Freeze protocol/scripts,
v14 manifest/plan/results/audit, module receipt/SO, baseline input files and event tensors
before GPU. Audit every crop/full-array hash and metric independently on CPU, check constant
white color/known radii and all v14 G hashes. No pixel p-values. No full-scene rerender.
