# v26 shared baseline/GT moments for both full and local FP64 scoring

2026-10-21. Authorized fixed-input mathematical/small CUDA diagnostics. No training,
backward,accepted edit,new data or novelty claim. All32 images/results already known.

v25 repeatedly filtered baseline and GT for every candidate. Strengthen BOTH complete
and local score paths:prepare a per-view immutable FP64 cache of GT mean,GT variance
and baseline SSIM (three fields per RGB pixel). A candidate then filters only X,X^2,
X*GT. Keep v24 FP64 operations/weights/boundaries and fixed summation order;compile
without fast math or FMA contraction. Source-conditioned cache identity assumes fixed
baseline RGB,GT,camera/resolution and Gaussian weights. Caller must rebuild when any
changes. No cache validity or free reuse across training updates is assumed.

Prep computes five baseline/GT moment fields transiently and keeps3 fields;candidate
scratch has3 fields instead of8. Cache needs3*3HW*8 bytes. Do not call reduced scratch
a total-memory improvement without charging retained cache and construction.

Use v24 synthetic fixture inputs and all32 existing real images. Adapter checks each
cached score/map call against original v24 unshared GPU computation literally,plus
the unchanged independent CPU map/scalar tolerance gates. Empty/unchanged/duplicate
and border cases retained. Normal mode produces only Kx3 scores/stats. The same new
kernel runs full-frame and local rectangles. No tolerances are widened after results.
No native scene forwards required for this loss-module correctness stage.

After gates pass,freeze the same four candidate-score endpoints A(full scene),B(cached
full image),C(sequential packed),D(batch packed),but ALL use the SAME shared moment
cache. Fixed8 proposals across4 known views,2 warm blocks,8 balanced measured blocks.
Control budget derived as4*[1+(2+8)*(3*8+1)]=1004;no training iterations. Include dynamic
child forward,ROI planning/CPU sync/layout transfers,patch emission,trial moments and
all allocation in C/D. Same prepared proposal/model/GT exclusions as v25. No paired
cross-run comparison;new same-run A/D,B/C,C/D,B/D medians are primary.

Measure and persist separate baseline rendering/cache S_R and shared loss prep S_L,
as well as joint S_total=S_R+S_L. Record retained geometry/original-output/GT and shared
loss cache bytes separately. If original RGB is provided without caches,full scoring
pays S_L and local scoring pays S_R+S_L,so conditional ratio is(S_L+F)/(S_total+D).
If both require original rendering,use common S_total:(S_total+F)/(S_total+D).
These are same-process setup+warm-median arithmetic,not measured cold starts. Common
S_R includes geometry/alpha unpack that full RGB-only initialization could omit.
Also primary warm comparison assumes both caches ready. Invalidation/setup matters.

Preserve every block and startup/setup/end telemetry. Prior WSL SM readings were
inconsistent with reported maxima;do not assert stable clocks or reset/lock clocks.
Per-view direct wall endpoints remain measured observations,not training speed,
population confidence,independent-scene or final model-quality evidence.
