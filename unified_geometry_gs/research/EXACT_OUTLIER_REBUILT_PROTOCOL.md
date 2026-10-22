# Rebuilt-binary outlier capture v2: new diagnostic scope

Posthoc, before v2 execution. The original v1 binary-identity check remains
failed, and its archive/logs are preserved. Restoring original ARCH12.0 produced
source/build.ninja bytes identical to frozen24 (17 recorded files per branch),
but did not restore either old .so SHA. No original binary restoration is claimed.

Pin each current new .so hash alongside the frozen old hash in a separate
`pinned_rebuilt_binaries.json` and manifest before loading. Verify all 17 original
generated-file hashes first. Directly import those pinned .so files, without
cpp_extension/Ninja or recompiling either old cached backend; verify new binary
hash before and after import, and again after capture. All commands explicitly
set ARCH12.0, CUDA12.8, gcc11/g++11 and MAX_JOBS2. The new small probe is built
inside fresh v2 and cannot overwrite the pinned backends.

This scope is a new diagnostic on rebuilt binaries, not relaxation of the old
identity requirement. Require actual FP32 bit-pattern equality of complete
depth and slope images against frozen24 for **all 8 affected views**, both arms.
Also require FP32 bit-pattern equality of RGB/alpha/normal with archived
affine-base-plus-arm-delta reconstructions. If delta is exactly zero, preserve
the base bits directly (including signed zero). The original full RGB outputs
of B/C were not separately archived, so this check's target is explicitly the
available reconstruction. Any image mismatch terminates v2 before that view's
ray data are accepted; no new tolerances or pixel substitutions are permitted.

Then capture all 9 selected rays using the unchanged buffer/layout, actual
n_contrib, whole-tile candidate retention and copied-expression coefficient/
initialization probe described in EXACT_OUTLIER_CAPTURE_PROTOCOL.md. The selection
and coefficient-array schema stay unchanged. Source/protocol/model/binary hashes,
generated probe bytes, layouts and each ray are archived atomically.

Even complete output equality does not uniquely determine historical hidden
buffers or machine instructions. Captured buffers are from the pinned rebuilt
binary execution, not the lost historical binary. Copied coefficient/eligibility/
initialization expressions are still diagnostic replay values, not original
instruction instrumentation. No root-cause, root-error, quality or speed claim
is made by this capture; independent high-precision/FP32 root oracles follow.

Execution correction before v3: v2 verified both affected scan37 images and
captured rays00/01, then failed because a new RGB reconstruction tensor used
the local variable name `expected`, overwriting the model-SHA string before
the post-scene assertion. Preserve v2 source archive/failure/data. Rename that
image variable; check the actual model against the separately stored manifest
SHA; rerun the same full9/8-view protocol in fresh v3. No scientific threshold,
selection, binary pin or output-equality gate changes.
