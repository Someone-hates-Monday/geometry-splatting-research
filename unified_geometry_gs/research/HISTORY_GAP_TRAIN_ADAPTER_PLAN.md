# Next stage: actual sampled GPU regularizer and zero-weight controls

Start from HISTORY_GAP_WARP_RESULT.md and results/history_gap_warp_checkpoint_v1.json.
Warp all891+64+12special+2absent passes the independent numerical gates;
full GPU loss/raw batch11.993ms vs same-process serial79.119ms. This is
not training performance/quality. Current warp .so resides in the failed
FIRST runner's generated directory: numerical kernel is valid and reused
by completed warp_v2. Pin exact path/SHA from warp_v2 manifest, not its
directory name/status. Keep CUDA/graphV2 sources frozen and use new files.

Now implement a research adapter to live GaussianModel raw parameters;
do not merely optimize cached per-occurrence variables and call it training.
Use actual shared IDs, gather _xyz/_scaling/_rotation/_opacity before
Float64 conversion, with filter_3D held fixed. Use actual activated
Float32 means, filtered sigma, normalized quaternion, per-primitive AMP
and actual query xy captured at the native render site. Source layout is
36 columns from benchmark_csr_decode.src_rows, but assemble sampled J
rows directly on GPU instead of full P rows/CPU arrays. Gradients must
scatter back to the shared original Float32 parameters.

Read exact callable signatures from existing helpers:
- benchmark_csr_decode.py run: actual query/AMP private capture and fields.
- benchmark_csr_capture.py offsets_for plus native count/capture APIs.
- exact_outlier_capture_rebuilt.native_args and parse provide actual native
  activated inputs/layout. Use pinned .so metadata from the completed
  csr_decode_v1/csr_capture_v1/full_topology_v1 manifests.
Full CSR can be a first smoke baseline, but count/capture/validation,
sampling/packing/preparation/transfers and raw backward must all be timed.
Do not copy entire rays/nodes to CPU in an iteration and exclude that
cost. Do not approximate query xy from an unverified pixel formula.

Empty contributor rays need an explicit documented definition/handling;
the current kernel rejects them. Numerical negative statuses/NaN must
abort a diagnostic before optimizer updates, never silently reward zero.
The tail branch includes both finite-after-last and absent roots; its
status is not a mathematical absence certificate.

Integration can use a new research wrapper around train.py, preserving
production sources. Existing train_solver_branch.py/checkpoint_state.py
implement bound optimizer restoration and camera/RNG sidecars;
train_holdout_branch.py shows prior fixed train/test splits. train.py's
actual loss.backward anchor is currently line666; verify source/hash and
frame names instead of assuming a line in a new checkout. Existing GS
has extra appearance/SG fields: regularizer should only add to the intended
11 raw variables; test all other parameter groups for unchanged behavior.

Fix a smoke protocol before results: disabled vs zero-weight control,
renderer output replay, shared-ID scatter equivalence to the existing MP
subset, finite gradients and complete optimizer-state binding, deterministic
camera/sample schedule independent of training RNG, a few nonzero updates,
complete iteration cost. Existing renderer atomics can be nondeterministic;
measure control repeats rather than relabel their drift as method effect.
Do not start a full quality run until these pass. Then choose predefined
weights/budgets and paired controls; compare GT, held-out images and model
capacity with all scene/evaluator seeds labeled correctly. Original147-view
GT and all previous failures remain unchanged. Overall goal is active.
