# Locality and stable merge for a cached TGS subdivision query

2026-10-21. Source-conditioned correctness argument, not a new compositing theorem.

Fix a camera, effective external primitive parameters/colors and deterministic per-pixel
alpha/compositing recurrence. One parent is removed and four children are appended.
Assume preprocessing of each unchanged external primitive returns the same defined
cache fields, and the four separately preprocessed children return the fields they
would have in the complete changed scene. Invisible cache fields need not be defined.

In the pinned native implementation, each visible primitive emits a key per tile in its
cached rectangle, with key(tile,positive FP32 center-depth bits). Prefix scan positions
put emissions in primitive-index order. A stable key sort therefore puts unchanged
externals before appended children at equal depth. NVIDIA's DeviceRadixSort documentation
specifies stability:
[source](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_radix_sort.cuh).

For each tile, remove the parent from the old sorted list, then merge that list with
the sorted four-child list, choosing old external first at equal keys. This returns
the identical full changed-scene tile list after canonical ID mapping. No primitive is
dropped, truncated or duplicated. This is the ordinary stable sorted-list merge property.

Outside the union of parent/child tile rectangles, neither edited primitive list entry
exists. The unchanged tile list, alpha and colors produce unchanged output. Inside the
union, replaying the merged list under the same recurrence yields the full changed-scene
RGB,alpha,last/median ordinal, including support rejection, cap, cutoff and the native
skip-current stop. Thus copying old output outside and recomputing inside is sufficient.

This statement is exact-arithmetic/deterministic-recurrence conditional. Separate CUDA
compilation could change rounding/instruction choices; the implementation therefore
requires literal full-image gates, not an assumption that equal algebra implies equal
FP32. All32 frozen old cases check every merged tile list, visible cache field and full
RGB/alpha/path output against the complete native split. Auxiliary weighted-depth/normal/
distortion fields, per-primitive maximum blend and backward are not query outputs and
are not claimed interchangeable with a full training renderer.

Cost with a reusable original cache: child preprocessing plus its tile emissions/sort,
mask over image tiles, full output copies, and candidate replay in affected tiles. The
prototype also calls the complete four-child native renderer, paying for its image
allocation/rasterization, which is not optimized away. Baseline setup/storage, event
construction and multiview cache invalidation remain necessary practical costs.
After any accepted scene update the old cache must be rebuilt or updated correctly;
the same baseline cannot silently serve sequentially accepted mutations.

No novelty of stable merging, incremental rendering or render-once/local evaluation is
claimed. [GaussianPOP](https://arxiv.org/html/2602.06830) already exploits analytic local
error computation for deletion. The present implementation tests a specific exact-output
counterfactual subdivision query, including changed center keys and native stopping.
Its value as a density policy and its relation to broader incremental-rendering literature
still require investigation.
