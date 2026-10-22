# v17 frozen fixed-cache validation

2026-10-21, after the three development pixels. Existing public flowers/room7000,
v14 cameras0/1, event indices1/2 (both exclude development indices0/5). No training,
GT quality testing, backward, source edits or new CUDA build. At most12 full native
forwards: four original views plus eight held-color split recaptures. All five native
output array hashes must literally match sealed v14. Reuse sealed v15 trace module.

Freeze all input hashes and sample coordinates before any new GPU output. For each
of8 cases choose four quadrant centers of the existing v14 change rectangle if nonempty,
and deterministic SHA256-ranked global positions to make8 unique pixels. Empty rectangles
use eight global positions. The rectangle uses prior outcomes, so this is a directed
mechanism validation on known views, NOT held-out photographic or statistical confirmation.
Retain empty-support and zero-effect cases. No tolerances or new thresholds selected
from bound outcomes. Exactly64 paired endpoint pixels,256 tiny trace calls.

Each packet is traced with native skip-current stopping and without stopping. Actual
RGB/alpha/last ordinal must literally match the corresponding native full-image pixel.
No-stop and actual traces must agree on all shared alpha/support inputs. Retain all
packet rows. The theorem applies to exact values of these recorded CUDA effective alphas,
not a proof of preprocessing, __powf or real-space triangle geometry.

Effective external layers must agree after canonical ID mapping. Missing parent candidates
are represented by an explicit alpha-zero parent at position0 and its frozen held color;
this changes no exact composite and is a valid fixed-slot choice, even if children are
active. This extension is fixed before validation and not an adaptive choice for tightness.
Premise failures are recorded as uncertified and retained, not discarded. Include real
early-stop cases if present, but do not claim their existence without checking.

Audit equation(1) by exact rational arithmetic. Add the exact omitted-tail sum and the
conditional normal-range gamma rounding envelope from the mathematical note. Check each
RGB channel against native endpoint differences. Record all zero errors, premise coverage,
nonzero bound/error ratios, early-stop counts, and absolute bounds. Ratios are descriptive;
no p-values, GT gain, efficiency or training benefit claim follows. Successful development
checks do not authorize declaring the method useful; loose bounds can reject the approach.
