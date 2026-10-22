# Undefined ratios in original distance-stress summaries

The original stress process exited0 and retained all 4608 finite per-case raw RMS outcomes.
Some selected primitives have constant appearance and hence zero copying error. Its unguarded
summary ratios divided by zero and emitted Infinity/NaN. Those ratios are undefined; they
must not be interpreted or used in the report. Original results/script/warnings are retained.

`audit_tgs_sh_radius_v13.py` independently recomputes every raw stress RMS using the pinned
polynomial SH evaluator instead of the original SciPy evaluator. It writes a new separate
strict-JSON audit/summary, with copy RMS<=1e-14 ratios explicitly undefined and counted.
All raw RMS and worse-than-copy counts keep every primitive; only ratio summaries omit
undefined denominators. This is a post-hoc reporting repair, not a changed outcome threshold,
new sample, refit or replacement for the original stress. Both versions are sealed.
