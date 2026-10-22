# v29: equal-budget joint edit selection with two-view structural coverage

Written after v28's mixed/tiny transfer results. Exploratory next design, not a
confirmation or a revision of v28. Public flowers/room 7k checkpoints remain fixed.
Only small forward CUDA diagnostics, mathematics, data identity and CPU evaluation
are authorized here. No backward, optimizer, training, model acceptance or production
change. The requesting user owns the research; no project material is uploaded.

## Frozen views and candidate universe

Reuse precisely the four v28 calibration images/cameras and original source identity.
Choose two NEW test cameras per scene, from sorted cameras[::8], excluding every v10,
v12 and v28 selected camera. Ascending SHA256('tgs_joint_policy_v29/scene/img_name'),
then name, chooses the first two, before any new payload/output. They are untouched
relative to this project's preceding probes, not independent scenes or independent
checkpoint-training experiments. Freeze exact JPEG and published test GT ZIP members.
Fetch them only after choices and calibration joint gates pass; original JPEG and
published PNG must decode identically, with no crop, registration or rescaling.
Public downloads are strict bounded HTTP Range/206/Content-Range, CRC/hash-checked,
TLS verification on, total new cached payload<=96MiB, each request<=16MiB. Windows
native TLS is used for published GT because v28 documented WSL transport failures.
One attempt per missing range; preserve failures and use a new addendum for recovery.

For each scene choose256 distinct parents using only calibration geometry. Require
opacity>=.01, native parent radius>0 in BOTH calibration views, positive native tile
rectangle areas, and maximum area<=one quarter of the image. Exclude the old v20
four stress IDs and all32 v28 candidate IDs. Rank eligible IDs by (MINIMUM of their
two calibration tile rectangle areas, ID), split into four equal-count integer bins,
and choose64 per bin by ascending SHA256('tgs_joint_policy_v29/scene/parent_id'),ID.
If any bin has fewer than64, stop; do not resample. Save all eligible IDs and both
area measures. Structural visibility does not guarantee actual RGB contribution.

## Candidate, loss and finite-precision contract

Official four midpoint children inherit sigma/opacity/SH. For each camera original
FP32 colors are evaluated once and kept exactly for unedited primitives; each set
of four children uses the existing child-only degree3 physical-direction Python SH
evaluation. This explicitly carries v28 r1 forward. Do not reevaluate all original
colors after changing tensor shape, or claim equivalence to a pipeline that does.

Use existing v22 planner, v25 packed RGB replay, v26 shared FP64 image-loss CUDA
modules; no new CUDA source/build. Score eight candidates per batch. Retain all256
per-view MSE/L1/SSIM deltas, perturbation energies, structural ROIs, patch hashes and
actual RGB-change flags. This does not promise full native auditing of all unaries.
Equal-weight means over the two calibration views define candidate score. Primary
image form:0.8*L1+0.2*(1-SSIM), negative change improves. This excludes global mean
opacity/size/normal/distortion terms and is not confirmed historical training loss.

## Frozen joint selection and references

Budget is EIGHT midpoint edits (+24 triangles) for each of five forced comparators:
lowest mean signed mixed change; lowest mean signed MSE change; lowest mean squared
render perturbation; largest MINIMUM calibration native parent rectangle area; and
fixed hash. Each rule greedily visits its sorted candidates, ID as numeric tie breaker,
and accepts the next candidate whose ROI expanded5pixels (SSIM output support) is
disjoint from all previously chosen expanded ROIs in BOTH calibration views. Stop at
eight. This is a deterministic greedy rule, not an optimal independent-set solver.
If any rule cannot fill eight, report inability; do not reduce budget or change pool.
All comparators share candidates, conflict constraints and budget. Also report no-edit
and a guard that abstains unless the primary predicted joint mean is below-1e-13.

Freeze scene choices before new full joint render outcomes, and the global choices
before any new heldout payload. Canonical joint order: delete selected original IDs
together, keep all other original order, append four children per ascending parent ID.
Reemit selected unary patches; hashes must match their original scoring patches.
For calibration, reassemble the full predicted joint mosaic from those patches and
require actual joint native RGB to match literally. The v27 sufficient locality
theorem applies under this contract; overlap at unobserved test cameras is allowed
and tested by actual joint rendering, never assumed additive there.

Each distinct selected set is rendered once per camera; identical sets are reused
across rules. For two scenes:4 calibration baselines+4 child banks+at most20 calibration
joint references+4 new test baselines+at most20 test joint references = at most52
new native forwards. Candidate replays/loss kernels are counted separately from
native calls. Freeze input/source/module hashes; unchanged model input hashes must
hold. Stop on failed bitwise gate, preserve all partial/failure artifacts.

## Outcomes, audit and inference boundary

Save full baselines, joint image changes as exact RGB patches with full RGB hashes,
full shared-FP64 joint scores and all selection inputs. Independently reconstruct
every joint image on CPU, verify full hash, recompute raw MSE/L1/SSIM and PSNR changes.
Scalar tolerance:abs error<=1e-13+1e-9*abs(CPU reference); no tolerance rescues a failed
RGB equality. Calibration unary-sum and actual joint scores must agree within this
recorded reduction allowance, separately from the literal mosaic prerequisite.
Metrics on untouched views are actual joint-native references; no test-oracle or
test-based choice changes. Show both test views and all forced comparators, guard,
and no-edit. Count actual RGB-changed views; do not remove zero/unfavorable cases.

Question:does this fixed, shared-coverage, equal-budget diagnostic give meaningful
heldout benefit for signed-loss selection? Falsifiers:benefit vanishes/reverses,
effects remain negligible, or competing rules match/exceed it. Rivals:limited view
coverage despite structural eligibility, occlusion, selection overfitting, tiny edit
budget relative to model, loss tradeoffs and useful capacity requiring adaptation.
Changing v28 pool/coverage/budget together means v28-v29 differences cannot causally
isolate any one of those factors. These are TWO existing scenes; no pixel/candidate
p-values, population confidence intervals, final trained-quality or geometry claims.
Any next adaptation experiment requires separate human authorization; this protocol
does not grant it. Scientific novelty and the overall research goal remain unproven.
