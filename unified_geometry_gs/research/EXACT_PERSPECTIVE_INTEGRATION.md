# Exact perspective depth integration v1

Protocol and scope, recorded before compilation/smoke. This is inference-only.
It implements a complete image **median-depth functional under legacy affine
selection**, not a complete perspective-correct renderer or trainable method.

Call graph: `gaussian_renderer.render` -> Python GaussianRasterizer autograd
forward -> `_C.rasterize_gaussians` -> RasterizeGaussiansCUDA ->
CudaRasterizer::Rasterizer::forward -> production preprocess, tile duplication,
depth sort -> FORWARD::render -> renderCUDA first compositing pass -> median
subdivision iterations -> packed median-depth/interface-slope image.

The current initial pass computes RGB, alpha, normal, contributor cutoff, and
root initialization from affine conic/planes. All of these remain unchanged.
The same selected IDs, affine power/alpha eligibility, .99 cap, affine prefix
cutoff, order, SAMPLE_RANGE, initial bracket, SPLIT, five iterations, finite
root validity gate, and view-normalized output convention apply to all arms.
Only p, reciprocal sigma, and alpha used **inside median T evaluations** change.
Exact alpha receives the same .99 cap; no second alpha-threshold filtering.
Consequently an input with affine eligibility and tiny exact alpha still stays.

Arms:

- `affine`: generated copy of unchanged production rasterizer source.
- `exact_packet`: recover coefficients from actual FP32 conic/plane/screen-UV
  packet using FP64 Schur arithmetic per ray, then cast coefficients to FP32
  for the unchanged subdivision solver. Screen projection epsilon and clipped
  geometry reconstruction are included, as in production_packet_v3.
- `exact_factor`: independently precompute per-primitive camera-space inverse
  factors from filtered scales, activated quaternion, and view matrix; evaluate
  each ray directly in factor space. Store 9 factor and 3 camera-mean FP32 values,
  use FP64 scalar evaluation and cast to the same FP32 solver. This reference
  retains actual activated FP32 rotation/view errors; it is not high-precision
  raw-PLY ground truth and does not certify packet/direct equality universally.
  Its additional 48 bytes/primitive and preprocessing kernel must be included
  in subsequent whole-render timing/memory comparisons.

The generated copies live separately; production files and installed extension
are unchanged. Inference proxy rejects calls when torch grad is enabled and
rejects every backward, integrate/SDF and sampled-depth method. These other
entry points still compile their unchanged code for link compatibility, but
cannot be accessed through this backend proxy. The original normal image remains
affine; no consistency claim between it and the modified depth is made. NCC
training uses integrate/sample paths and backward not covered here. No training
is allowed until those paths and complete VJPs have separate designs and tests.

Predeclared one-view smoke: scan37/camera0000, saved resolution2 model30000,
kernel0, all three arms; installed renderer also as control. Save all image
outputs, source/generated/binary hashes and CUDA/compiler versions. Require
finite RGB/alpha/normal/depth/slope, radii exact equality, and RGB/alpha/normal
bitwise equality across generated arms; inspect generated affine vs installed
parity without silently treating any mismatch as acceptable. Report all
depth validity-mask disagreements, absolute differences, and packet vs factor
relative depth errors (floor1e-4). Packet/factor gate: zero invalid values,
same finite-root mask, and max relative <=1e-3. Gate failure is preserved and
diagnosed, not fixed by selecting pixels or widening tolerances. No GT quality
comparison, benchmark or training will run in this initial smoke.

Numerical limits: general FP32 packet recovery already has retained synthetic
failures; local 1992-row production passing does not cover every image input.
If Schur data are nonpositive/nonfinite, mark the pixel depth invalid and write
NaN interface slope for diagnosis, with no affine fallback. Existing finite
camera-origin normalization remains absent from this production functional.

Follow-up requirements before a full method: conservative exact support-domain
design, exact initial RGB/alpha/normal semantics, sampled/integrated NCC forward,
all raw-parameter and accumulated CUDA backward paths, independent root/gradient
checks, fixed-model GT quality and full timing, then a predeclared training gate.
