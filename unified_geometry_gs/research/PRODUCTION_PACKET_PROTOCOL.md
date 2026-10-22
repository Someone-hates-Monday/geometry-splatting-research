# Production FP32 packet screen v1

Frozen before first run. This follows the failed rounded-double packet screen;
it cannot repair or replace that screen's 1/128 synthetic failure.

Question: does the exact-real clipped Schur recovery remain accurate when its
inputs come from the actual FP32 production preprocessing operations, rather
than rounding a packet computed in double precision?

Use the unchanged 24 rays and every one of their 1992 selected primitive IDs
from `experiments/clipped_packet_v1`. Preserve all 46 clipped cases and every
culling/nonfinite outcome; no replacement selection, fallback, or training.
Load original scenes and Camera objects at saved resolution. Use production
GaussianModel getters for filtered scales/opacities and normalized quaternions.

Execution is a diagnostic extension, **not the installed renderer binary**:
extract computeColorFromSHSG, computeCov2D, and preprocessCUDA verbatim from
the current production render_forward.cu and directly launch the copied full
preprocessCUDA<3>. Its original frustum/tile culling, projection denominator
epsilon, matrix products, conic inverse, plane, opacity coefficient, and reciprocal
width are preserved. Precomputed dummy RGB bypasses only SH color evaluation.
Archive original/generated sources and compiler commands/binary hashes. Use
production `-O3 --use_fast_math` flags. FMA defaults and optimization contexts
may differ from the installed binary; this is not binary equivalence proof.

Primary packet uses saved production screen UV to reconstruct qmean, clip that
ratio for qbar, and original pixel/intrinsics for qray. Decode once with NumPy
float64 and once with the existing Torch float32 recovery on CUDA (separate
elementwise kernels, not a fused production renderer implementation).
Compare to the same archived FP64 world coefficients. For each input require
finite values, positive Schur determinant and denominator, nonnegative energy,
peak relative error <=1e-5 with floor 1e-8, width relative <=1e-4 with floor
1e-8, and alpha absolute <=1e-4. Cull counts are failure counts. Preserve the
older screen's thresholds and all per-row errors.

Prespecified diagnostic arms, with the same thresholds and all rows: replace
qmean/qbar with the extra probe output from production transformPoint4x3 and
the exact same clamp/multiply/divide sequence (not stored production packet);
or replace qmean/qbar with archived double camera ratios. This distinguishes
screen projection/geometry encoding from other FP32 preprocessing and covariance
inverse inconsistency. These arms use additional unavailable information and
are not proposed deployable fixes. Record UV-vs-camera ratio difference and
packet differences from archived double. No image/geometry quality, runtime
gain, VJP, synthetic coverage, or novelty conclusion is permitted.

Because the same primitive may occur in several rays, 1992 rows are diagnostic
inputs, not independent samples. Scene/ray IDs are already development data.
