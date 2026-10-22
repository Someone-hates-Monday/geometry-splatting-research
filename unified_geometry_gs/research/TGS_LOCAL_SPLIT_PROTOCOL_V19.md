# v19 cached single-parent subdivision query

2026-10-21. Fixed-input implementation/diagnosis only; no optimizer, training or backward.
This is not yet a novel densification policy or a production renderer replacement.

Reuse one original native scene's geometry and sorted tile lists. Obtain four children
using official midpoint subdivision and held parent RGB. Preprocess them using an
unchanged native4-primitive forward. This first reference also pays for that forward's
full-image allocation/rendering, explicitly included in later query cost; do not hide it.

The changed pixels are contained in the union of the cached parent/child tile rectangles.
For each affected tile merge the original list (excluding the parent) with the four-child
list by the original positive FP32 center-depth bit key. At equal keys existing external
primitives precede appended children; preserve child stable order. Recompute original
window/cutoff/cap and skip-current stopping. Clone baseline RGB,alpha,paths and update
all pixels in affected tiles. No fixed K, hit truncation or reuse of trial split outputs.

Validate all32 old v14 event/camera cases, including all invisible/zero-effect cases:
2 models,2 old cameras/model,8 fixed events/camera. Before any new outputs freeze scripts,
native binary, protocol, model/camera and sealed v14 references. Cap68 native full forwards:
4 baselines,32 four-child forwards,32 full-scene split references. Four-child forwards are
part of the algorithm; full-scene trial references are used only to assess correctness.

Gates: baseline and full reference all5 output arrays must literally reproduce v14 hashes.
All defined external geometry cache fields and child cache fields (depth,window,opacity,
rectangles,radii; visible only) must match original/4-child caches respectively. Independently
compare every merged tile list with the full native split list, including ties and empty
tiles. Global RGB,alpha,last/median ordinal must match full reference literally; reject
even tiny mismatches. Outside affected tiles original values must remain unchanged.
All32 cases retained; no picking visible successes or changing tolerances after a failure.

Observe counts/affected coverage only in correctness phase. Timing is prohibited until
all gates pass; then freeze a separate balanced paired protocol before timings. Report
baseline cache setup/storage, event definition preparation exclusions, trial native
preprocessing included in local query, output cloning, synchronization and amortization.
Do not interpret cached single-event query acceleration as training/inference speedup.
No GT or geometry-quality improvement follows from matching the original split image.
