# Exact integer counterfactual image-error reduction

2026-10-21. Extends the source-conditioned v19 locality/merge argument. No theorem about
GT generalization,optimization convergence or full CUDA compiler semantics.

For a fixed proposal p, let X be its full native RGB image and B the original image.
Let q be the specified two-RN-op PNG8 quantizer and G the fixed decoded GT bytes.
The score is D_p=sum_pixels,channels[(q(X)-G)^2-(q(B)-G)^2]. Outside the union of edited
primitive tile rectangles, the native primitive list is unchanged and X=B; those terms
are zero. Therefore evaluating only this region yields the full-image score exactly,
provided the local replay produces the same RGB values/quantization.

Preprocessing fields of a primitive depend on its own inputs/camera,not on unrelated
counterfactual primitives. Thus the four children for each proposal can be preprocessed
in one concatenated bank,under this premise. The bank's rendered RGB is NOT a proposal
image and is discarded. Each proposal independently reconstructs its four member IDs
per tile using native rectangles and stable depth bits,then merges them with old
external records while removing only its own parent. No other proposal's children
participate. This preserves the complete split list under the same conditions as v19.
FP32 field/list/global image equality is explicitly checked against single full renders;
algebra alone is not treated as a bitwise proof.

For q(X), each integer lies in[0,255]. One channel's squared error is0..65025,so each
channel difference is within[-65025,65025]. Each256-pixel block's three-channel sum is
within +/-49,939,200,safely inside int32. Any global partial sum's absolute value is
bounded by3*H*W*65025. The wrapper restricts3*H*W<=INT_MAX,which is far below int64
overflow at this bound. Integer atomics therefore have order-independent exact sums;
unsigned two's-complement atomic representation gives the same signed int64 result.

This avoids floating reduction tolerance and a full output image per proposal. It does
not make the RGB recurrence exact in real arithmetic,replace the native float L1+DSSIM
training objective,or validate a split-selection policy. A negative score is only the
measured fixed-view candidate change. All proposals/GT views are inherited known inputs,
not a new independent quality benchmark. No original scene mutation is accepted here.
