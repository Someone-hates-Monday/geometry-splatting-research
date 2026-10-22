# Input-reader repair, before target-field observations

2026-10-21. Original `probe_tgs_sh_real_fields_v13.py run` exited1 (exec chunk901357)
at `.numpy()` because saved features require gradients. All predefined basis/Gram checks
completed; flowers primitive IDs were saved; no coefficient NPY, target colors, fits, quality
measurements or room selection had been produced. Preserve the failed script/plan/selection.

Use a derivative v2 driver and a new `real_fields_v2` output directory. The sole data-path
repair is `.detach().numpy()` rather than `.numpy()` for readonly features. The v2 driver also
checks the original source hash and verifies flowers IDs against the retained old selection.
No parameters, sample rule, shell, degree, quadrature, metric or threshold changes. This
is a confirmed terminal error, not an observation timeout or reason to duplicate training.
