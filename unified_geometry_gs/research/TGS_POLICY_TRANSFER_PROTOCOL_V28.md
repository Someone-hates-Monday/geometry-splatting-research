# v28: frozen zero-step edit selection and untouched-view transfer pilot

Scope: public flowers/room 7k checkpoints, fixed renderer and coherent inherited
SH parameters, no training/backward/optimizer or accepted model edits. This is a
small diagnostic of immediate edit utility, not final trained-model quality or a
novelty claim. It follows the v27 negative interaction result. Accountable research
owner is the requesting user; all calculations remain local, public image ranges
are downloaded without uploading project material.

## Frozen sampling and labels

For each scene sort all archived cameras by img_name. Indices divisible by eight
are the archived test split; others are training cameras. Exclude every v10 selected
camera and v12 selected camera. Within each split take the first two by ascending
SHA256('tgs_policy_transfer_v28/scene/split/img_name'). Training-split views are
calibration only, test-split views are evaluation only. They are new views relative
to our preceding probes, not independent scenes or independent checkpoint training.
This is a post-exploration protocol, not a registered confirmatory study.

Freeze all camera/member identities before payloads. Fetch four calibration JPEGs
first. Freeze selected edits before fetching four held-out JPEGs and their published
GT PNGs. Calibration GT is the official already-downsampled source JPEG, decoded to
RGB with no resizing; there is no published train GT in the evaluation ZIP. Each
calibration image must have the checkpoint's documented output dimensions. Test
JPEG and published PNG must match exactly as decoded RGB, camera index and shape.
Any missing/failed identity is reported, not replaced by a favorable image.
The total new cached HTTP Range payload across all phases is at most96MiB;
per-request limit16MiB, strict206/Content-Range/CRC/hash checks, no whole ZIP fallback.

## Candidate universe and scoring

Each scene has32 distinct parent candidates, selected before image-loss outcomes
using ONLY calibration preprocessing. Require effective opacity>=.01, visible in
at least one calibration view, positive maximum native tile rectangle area, and
maximum area<=one quarter of the image. Exclude all earlier v20 four stress parents.
Sort eligible IDs by (maximum calibration rectangle area, ID), partition their
ordered list into four equal-count bins using integer endpoints, and take eight
per bin by SHA256('tgs_policy_transfer_v28/scene/parent_id'), ID as final tie breaker.
If a bin has fewer than eight, stop without replacing the protocol. Save every
eligible ID and area. This pool is geometry-stratified, not population-random.

Every edit replaces one parent by its four official midpoint children, inheriting
sigma/opacity/SH; child colors are recomputed at physical child centroids for each
camera with the same existing degree3 Python SH convention. No parameter fitting.
Use cached native semantics and existing v22 planner/v25 packed replay/v26 shared
FP64 image-loss module. Process eight candidates per batch to bound temporary memory.
The candidate pool is shared by every comparator. No test geometry or GT selects it.

Scores are the mean of two calibration views' RAW normalized squared-loss changes,
L1 changes and SSIM changes. Primary policy: lowest mean(.8*Delta_L1-.2*Delta_SSIM),
forced to choose ONE candidate; separately report a guard that abstains unless this
mean is below -1e-13. Additional one-edit comparators: lowest raw MSE change;
lowest mean squared render perturbation ||trial-base||^2/(3HW) (an elementary
perturbation proxy, NOT a faithful GaussianPOP implementation); largest max native
calibration rectangle area; and lowest frozen parent hash. Ties use parent ID.
All5 forced policies add exactly3 triangles; no-edit is the zero-change reference.
Freeze choices and their complete32-row calibration score table before test fetch.

## Measurements, gates and inference

For all32 candidates on each of two test views, compute the same fixed scores after
choices are frozen. Never tune choices on test outcomes. Render each DISTINCT chosen
candidate with a full native reference on every calibration/test view, compare full
RGB bits to the reconstructed cached patch output, verify unchanged input hashes,
and independently recompute full-image CPU MSE/L1/SSIM differences from saved RGB
patches. Scalar gate remains abs error<=1e-13+1e-9*abs(reference); no tolerance may
rescue failed RGB equality. Baselines and selected full-reference patches are saved;
all candidate per-view scores, perturbations, ROIs and reconstructed full RGB hashes
are retained even for unfavorable outcomes. Record actual native-call count; expected
at most8 baseline+8 child-bank+40 selected-reference=56 calls across both scenes.

Primary observable: held-out mean image-form change for the calibration-selected
primary policy, with both views separately displayed, no-edit and equal-count
comparators. Negative values improve. Also report MSE/L1/SSIM separately, per-scene
PSNR change from raw MSE, calibration-to-test direction changes, guard abstention,
and descriptive test-pool rank/regret. Test-oracle quantities are post-selection
diagnostics, never an implementable policy. PSNR is derived, SSIM is raw/unclipped;
neither is claimed identical to historical TGS reported quality. Checkpoint actual
L1/L2 branch/config remains unknown; image-form scores exclude global regularizers.

Prediction under test: loss-aligned calibration selection transfers better than
geometry/hash/perturbation comparators on held-out views in these two fixed scenes.
Falsifier: its improvement disappears/reverses or comparators perform as well/better.
Rivals: view overfitting, SH convention/checkpoint mismatch, immediate topology
perturbation rejecting capacity useful after adaptation, restricted pool, and small
effect/reduction noise. Two scenes are the reporting units; no pixel/candidate-based
p-values, population confidence interval, superiority claim from a few observations,
final geometry claim, or parameter-adaptation conclusion. Report negative/mixed
outcomes without a search for alternative thresholds or candidate seeds.

Protocol/source/input hashes are frozen per stage before new outcomes. Failures are
preserved; a failed stage is not silently overwritten/restarted. Test fetch depends
on a completed calibration choice receipt, test scoring depends on identity gates.
No new native CUDA code/build, timing benchmark, dataset-wide quality claim, training
or production implementation change is included.
