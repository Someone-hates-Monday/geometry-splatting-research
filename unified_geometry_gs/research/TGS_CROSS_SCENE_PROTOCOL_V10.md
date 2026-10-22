# v10 bounded cross-scene diagnostic, frozen before new model payload/output

2026-10-20. Human scope remains mathematical checks and small fixed-input CUDA diagnostics;
no optimizer, training, parameter tuning or GT quality evaluation. The full scientific goal remains.
Preserve v9 source/binaries/data/manifest read-only. Use its passing coarse and old v5 native64
oracle binaries without recompilation. Failed standard is not revived or measured.

## Question and rival explanations

Observation: v9 coarse preserved the specified oracle forward, and seven stump7000 cameras
had paired F+B wall ratios .734–.749 in one unlocked-clock session. This did not exceed the
numerically different historical piecewise implementation. New tests must address other scenes.

Candidate explanations, not findings: the outside-predicate benefit may persist across the
work distributions of new scenes; alternatively, a scene-dependent outside fraction, larger
preprocessing/cache cost, or different branch mix may remove it. Clock/order variation is a
measurement rival. Passing another scene does not establish a universal or novel method.
The first discriminating outcome is whole-image correctness, before any speed conclusion.

## Cohort and acquisition

Use the original public full_eval.zip listing and pinned official full_eval.py scene groups.
Choose exactly two NOT previously rendered TGS scenes, without observing their model/output:
minimum compressed checkpoint size among iteration_7000 outdoor scenes excluding stump,
and minimum compressed checkpoint size among iteration_7000 indoor scenes. Tie by archive path.
Listing predicts flowers7000 and room7000. Keep iteration7000 fixed; do not choose room30000
merely because its compressed file is smaller. No scene substitution after observing results.

Extract model, hyperparameters, cfg_args, cameras.json and the lexicographically first existing
test GT PNG per scene. The PNG is only a dimensions witness; its camera identity is not inferred
from its integer filename and no pixel/PSNR comparison is allowed on that assumption.
Use public HTTP Range, reject whole-archive fallback, <=650MiB total new network data per run,
<=16MiB per request, verify Content-Range/ZIP sizes/CRC and record SHA256. New ranges and
receipts go to a new source directory. Never execute downloaded checkpoint content; restricted
weights_only loader and the existing narrow known NumPy hyperparameter whitelist only.

Verify original source/hash and the expected saved tensor layout/activations/mask semantics.
Verify finite tensors, camera rotations, canonical sorted-name modulo8 eval split and serialized
test+train order. Real resolution comes from PNG header plus cfg image downsample rule, not an
extra downsampling of COLMAP focal length. Nonconforming metadata is a documented input failure.
Per scene select one test and one train camera by minimum SHA256('tgs_cross_scene_v10/'+scene+
'/'+split+'/'+name), before any rendering. Four full images, no camera substitution/tolerance change.
Record hashes of model/upstreams/cameras before/after GPU work. No synthetic relabeling as real data.

## Precision and proof checks

Run lean coarse and old native64 oracle, plus their counter twins, at each full selected camera.
RGB, all7aux, maxblend, radii, last/median contributors MUST be bitwise identical coarse/oracle;
coarse flags and all preflight arrays must be zero. Counter twins' forward/cache/path bitwise.
Both fixed dense RGB and weighted-depth upstream VJPs, all six groups finite; coarse/oracle
and diag/lean VJP relative<=1e-5 when reference L2>1e-10, else maxabs<=1e-6. Existing CPU
physical and FD evidence is inherited, not falsely counted as new independent tests.
All four images must pass to permit cost; preserve any failing image without narrowing cohort.
FP32 atomics are not required bitwise. No derivative-of-culling claim follows from this check.

Independently recapture old oracle anchors/rectangles/double coefficients, check matching caches,
all visible extents/domain, and exact Fraction E for deterministic512 SHA-ranked triangles plus
32 largest footprints/camera (deduplicate). The conditional proof remains scoped to cached c64.
Save forward/cache/path and all tested gradient arrays with hashes for independent reanalysis.
Independent audit recomputes all metrics from retained arrays before candidate acceptance.

## Optional cost, only after full independent precision pass

This protocol does not authorize a timing implementation until its separate balanced schedule,
raw measurement fields and analysis script hashes are frozen. Same-semantic coarse/native64
primary; historical piecewise context must remain explicitly different numerics. No timing failed
cohort. Retain all observations and describe unlocked-clock/session limitations. Do not transplant
v9's percentages to the two new scenes without measurement.

## Interpretation

Independent units for applicability are these purposively/transfer-size selected scenes, not
pixels, VJP elements, resampled blocks or four independent training seeds. A precision pass is
reference equivalence evidence, not GT accuracy. Cross-scene cost may strengthen an engineering
case, but geometry/image quality, novelty and training effects remain unresolved requirements.
