# v10 work completed and pending cost analysis

This is an INTERIM historical checkpoint. Final analysis, independent cost audit and figure
have now completed; use TGS_CROSS_SCENE_RESULT_2026_10_20.md as the current result report.

Update after this checkpoint: the live usage tool returned ordinaryUsageAllowed=true.
One retry of the SAME analyze command through the SAME require_escalated approval path was
approved and exited0. No execution channel was changed and no GPU experiment was repeated.
The analysis now exists; independent cost audit, final result report and sealing remain to finish.
The interruption section below preserves what was true at the time the first launch was rejected.

The broad scientific goal remains active and incomplete. Human authorization for this phase
was mathematics and small fixed-input CUDA diagnostics, without optimizer or full training.

## Verified completed work

- Retrieved flowers7000 and room7000 from the official public model archive using bounded
  Range requests. Ten members were CRC/SHA checked; successful payloads totaled608581345B.
  Initial download session71236 exited1 on the final PNG's header with RemoteDisconnected;
  session4335 safely skipped nine verified members and retrieved the missing PNG, then exited0.
- CPU audit session15753 exited0. flowers has938101 triangles at1256x828; room has2100000
  at1557x1038. Four cameras were selected before rendering: flowers test_DSC9128/train_DSC9169,
  room testDSCF4915/trainDSCF4697. Published PNG dimensions are used; no universal ceil rule
  is assumed for the images_4 folder. Original COLMAP FoV is retained.
- GPU precision session26656 exited0: four full images, unchanged v9 coarse and v5 oracle
  binaries plus counter twins, all six complete RGB/depth VJP groups saved and checked.
- First independent audit5073 exited1 due to comparison of invisible uninitialized anchors.
  Diagnostic65143 exited0 and established that only25910 radii0 flowers train anchor rows
  differed; no defined output or visible-cache mismatch occurred on any of the four recaptures.
  The original failure/script/progress/arrays remain retained.
- Separately frozen v2 audit81110 exited0. It checks projection anchors/rectangles only where
  the native source defines/consumes them (radii>0), retains all other complete output/VJP checks,
  and uses actual byte comparisons for recapture. All four images pass. Exact Fraction cache
  checks covered1632 edges per camera,6528 total. The original precision tolerances and cohort
  did not change. See TGS_CROSS_SCENE_AUDIT_REPAIR_V10.md and audit_repair_plan.json.
- Supplemental literal-byte audits47494 (v9) and17152 (v10) exited0 with zero differences:
  2948 old fields and88 new fields. A separate2592-pair FD sample audit also found zero sample
  bit differences, distinguishing signed zero. These do not claim all FD numeric errors<=1e-3.
- Cost benchmark session8172 exited0: all72 prespecified balanced blocks,216 native F+B
  calls, plus36 fixed warmups. All model/upstream/camera input hashes remained unchanged.
  No observations were removed and no timing rerun was performed.

## Pending work and actual interruption

The attempt to launch analyze_tgs_cross_scene_cost_v10.py was NOT executed. Automatic approval
review failed because of an account usage limit, with a message to try again at7:34PM. The
message explicitly says review could not complete; it is not a determination that the action
is unsafe. Do not bypass that check by switching execution paths. This is not a CUDA failure,
training failure, or a still-running benchmark. All previously observed GPU jobs are terminal.

The frozen analysis/audit/figure scripts exist, but cross-scene cost summaries have not been
computed or independently verified. Do not report a cross-scene speedup yet. v9 stump numbers
come from a separate session and cannot be substituted for flowers/room results.

After the approval condition is resolved, inspect the existing cost/results.json and frozen
cost_analysis_plan.json, run the original analysis followed by audit_tgs_cross_scene_cost_v10.py,
then generate and inspect the figure, write the result report, and seal the evidence. Do not
repeat acquisition, precision or benchmark merely because this analysis launch was rejected.
The v10 sources/outputs are not yet fully sealed; v9 remains sealed and untouched.

GT camera identity, actual image/geometry quality, novelty and training effects remain unproven.
This status is a checkpoint, not a completed academic result or a completion of the broad goal.
