# v22 batched fixed-state PNG8 error oracle

2026-10-21. No optimizer,training,accepted scene mutation,GT-driven proposal selection
or new density policy. Reuse all16 v20 H/M stress proposals and the two same calibrated
fresh views. Use already identity-verified v12 GT RGB bytes, with no resize/crop/fit.
Freeze scripts,CUDA source,model/camera/GT,proposal selection and control hashes before
new output. The score is PNG8 SSE change, not the native float L1+DSSIM training loss.

Pack each view's8 alternative proposals into32 child primitives. One unchanged v21
control forward obtains their independent preprocessing fields; its mixed-possibility
rendered image is discarded and NEVER interpreted as a counterfactual scene. The entire
forward,including its wasted allocation/render,will count in later batched cost. There
is no preprocessor-only backend or hidden free child cache in this version.

For each proposal, derive its union bounding rectangle from original parent/four child
native tile rectangles. Prefix-scan rectangle tile counts to schedule only those tiles;
skip rectangle holes with no edited member. At each tile form its four-child sorted
list directly from membership and positive depth bits,stable by child index. Merge with
the original external tile list excluding the parent,old external first at equal depth.
Evaluate unchanged window/alpha and skip-current native recurrence,with no fixed K cap.

Quantize RGB with TWO explicit RN FP32 operations x*255 then+.5,clamp[0,255],truncate.
Use int64 sum of [(trial8-GT8)^2-(original8-GT8)^2]. Each256-pixel block's sum fits signed
int32; global absolute partial-sum bound3*H*W*65025 fits signed int64. Signed atomic sums
are exact modulo2^64 and cannot overflow the representable mathematical result/partial
sum under this bound. No float reduction-order tolerance is allowed.

Before real data: check quantization around every8-bit half threshold,stable child/external
ties,multiple independent proposals,zero-ROI and deleted-parent-only cases,and257 original
layers. Software fixtures do not establish real-scene correctness.

Real gates: compare baseline PNG bytes against independent CPU v11 png8; verify GT JPEG,
PNG and saved decoded array identity. For each proposal,check batched4-child geometry
against the complete native/control split,each merged tile list against the full split,
and optional debug RGB/alpha/path output globally bit-for-bit. Debug uses the SAME kernel
with a runtime-selected output target; the normal API returns only scores. Native full
reference hashes must reproduce v20. Scores from the batch kernel and full-image GPU
reduction must exactly equal independent CPU int64 SSE differences for all16 proposals.
Cap20 native/control forwards:two original baselines,two32-child banks,16 full references.
Retain every score/sign and case; no model/quality/GT-generalization claim follows.

Only after gates pass,freeze timing on prepared proposals against full-scene+score and
sequential cached-query+score references. Include child forward,ROI plan/prefix scan,
scalar reduction and all temporary allocation. Keep baseline setup,packing exclusions,
GT/view limitations,stored-cache memory and mixed-scene child-render waste explicit.
No batch timing,novelty,policy usefulness or training benefit is assumed in advance.
