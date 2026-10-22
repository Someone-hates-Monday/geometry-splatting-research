# All full-image outlier rays: posthoc capture v1

This is a posthoc diagnostic after frozen_exact_render_v1 failed its unchanged
numerical and efficiency gates. Enumerate **all** common-positive packet/factor
pixels whose absolute view-normalized depth difference exceeds 1e-3 across the
24 saved image NPZs. Expect 9, retain every one, not just the prior eight largest.
Verify the prior run is complete and its sources/artifacts/binaries match.

For each affected frozen model/camera, rerun the unchanged packet and factor
inference native `rasterize_gaussians` forward under no_grad to obtain real
geometry/binning/image/tile buffers. This is a forward replay for capture, not
new training, GT evaluation, or repeated full24 timing. Require the entire
view's depth and slope outputs to be bitwise equal to their saved images;
preserve any mismatch as execution failure rather than treating it as data.

Parse only audited fromChunk prefix fields. Assert every actual buffer base
pointer is 128-byte aligned and each offset/view stays within actual numel.
Geometry: depth, ray_plane, optional exact_factor[12P], normal, clamped[3P],
internal_radii, UV, conic_opacity, RGB, tiles_touched, followed by CUB scratch
whose contents/layout are not inferred. Binning starts with sorted point_list.
Image starts with actual n_contrib. Tile starts with Gaussian ranges, followed
by max_contributor. Check all ranges fit actual num_rendered, IDs fit P, actual
n_contrib fits the tile range, and packet/factor tile lists/prefix counts and
unmodified packet fields agree. Retain **whole tile candidates**, including
entries beyond the actual pixel prefix and affine-ineligible entries.

A separate small CUDA probe includes the exact archived
`exact_perspective_ray.inc` functions verbatim. For each tile candidate record
legacy power, capped alpha, affine eligibility and peak/rho, plus both exact
p/rho/alpha/valid outputs. Original prefix membership comes from actual
img.n_contrib, not a reconstructed double-precision compositing cutoff.
The accepted mask is that original prefix intersected with the probe's copied
affine eligibility. Probe expressions/flags are copied; compiler context/FMA
may differ, so this is not a claim of original-kernel instruction identity.

Also replay the affine initialization in a one-thread CUDA probe, saving
mDepthinit, initial bracket, final affine T and contributor cutoff, plus rln.
These are **copied-expression probe replay values**, not otherwise unavailable
original internal state. Confirm the replay contributor count equals actual
n_contrib; do not silently substitute a mismatch. Raw original output only
contains view-normalized depth/slope, from which ray-depth is derived using the
recorded probe rln. No original final-bracket or endpoint trace is claimed.

Save selected primitive IDs and p/rho/alpha as FP32 arrays without decimal
roundtrips; save every candidate's UV, conic, plane, direct factor12, means,
activated scales/quaternions/opacities, raw mean/logscale/quaternion/logit/filter,
query/focal/camera matrices and pixel, actual n_contrib/ranges/max_contributor,
both original outputs, probe initialization and all excluded candidate masks.
All sources/generated/binaries/models and per-ray artifacts are hashed.
Manifest/progress/rows are saved atomically at each phase. Root will independently
analyze high-precision coefficient sums/roots and a separately labelled FP32
solver replay; capture itself makes no root-cause, quality or improvement claim.

Execution correction before v2: v1 ended at the original-binary SHA check,
before any ray capture. The launch omitted explicit TORCH_CUDA_ARCH_LIST, and
Torch rebuilt cache binaries with default 12.0+PTX instead of the frozen run's
explicit 12.0. Preserve v1 failure and archived source bytes. Restore both caches
using the original ARCH/CUDA12.8/gcc11 settings and require original binary SHA
identity before fresh-v2 capture. The updated entrypoint rejects any absent or
different ARCH before extension loading. This is an execution-environment
correction, with unchanged 9-ray selection, capture schema and scientific gates.
