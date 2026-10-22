# v10 audit correction: defined projection cache and literal bits

The first independent auditor (session5073, exit1) passed flowers test and then failed an
oracle-recapture comparison for flowers train. The original source, partial audit_progress.json,
failure traceback and all saved arrays are retained. No candidate implementation was changed.

The separately versioned diagnostic (session65143, exit0) recaptured ALL four fixed inputs.
It found25,910 differing anchor rows at flowers train; every one had radii==0. All visible anchor
rows, all rectangles, coefficient/error caches, radii, maxblend, contributor paths, RGB and seven
auxiliary outputs were identical. Other three recaptures had no field differences. Byte samples
and visible/invisible classifications are retained in recapture_diagnostic.json.

The native source establishes the meaningful mask independently of these observed values:
forward.cu initializes radii[idx]=0 and can return before points_xy_image[idx] is assigned;
points_xy_image is assigned only near the final successful preprocessing path. The binning
kernel in rasterizer_impl.cu emits keys only for radii[idx]>0. Backward preprocessing returns
unless radii[idx]>0. Invisible projection slots are therefore undefined scratch rather than an
observable renderer result or an error-envelope input. A whole-allocation anchor comparison
was an error in the auditor's scope, not a renderer failure that can be waived numerically.

The v2 auditor compares anchors and rect_min/max at the shared radii>0 mask, on every camera.
Radii themselves are still compared across ALL rows first; the mask cannot hide a visibility
difference. All other defined outputs remain checked in full. Full RGB/depth VJPs and exact
visible-domain/cache checks remain unchanged. Actual byte comparison strengthens the recapture
check. Original audit and GPU script hashes remain frozen in real_plan.json; new auditor hash,
original failure hash and diagnostic hash are separately frozen in audit_repair_plan.json before
the v2 run. Independent progress uses a new filename, preserving the original partial record.

A separate issue: np.array_equal treats signed zeros as numerically equal. The supplemental
literal-bit audits therefore compare stored bytes, including signed zeros. They cover all127
old v9 fixtures and seven old real cameras for coarse/oracle/counter fields, all four new v10
cameras, and the retained2592 paired FD endpoint samples. These are added provenance checks,
not modified GPU code, new tolerances, extra scientific samples or an alteration of old sealed
evidence. v9_literal_bit_audit.json, literal_bit_audit.json and v9_FD_sample_bit_audit.json all
report zero differing bytes/sample values in their respective comparisons.

Timing remains prohibited until the repaired independent four-camera audit passes, and all
literal-bit checks have passed. No scene/camera is removed and no floating-point precision
threshold is relaxed as part of this correction.
