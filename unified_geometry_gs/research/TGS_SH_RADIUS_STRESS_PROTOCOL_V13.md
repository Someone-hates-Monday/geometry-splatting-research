# Post-hoc distance-transfer sensitivity, frozen before stress outputs

2026-10-21. This follows inspection of the completed v13 shell results, so it is explicitly
post-hoc mechanism/sensitivity work. No new independent scene, camera capture or training.
Do not count it as confirmatory holdout evidence for a selected method.

Keep all 128 real coefficient sets, three axes and four original a values. Reuse the saved
degree3/4 fit32 coefficient conversions without refitting. At the same independent angular
grid, change camera radius to k*R, k in {.5,2,4}, keeping the world displacement delta fixed.
The reference appearance therefore uses normalize(d+(a/k)*axis). Compare the unchanged copy,
the frozen L3 fit and frozen L4 fit, raw RGB RMS. All 4608 cases retained. No parameter search.

Rival predictions: If shell projection is useful across arbitrary distances, frozen converted
coefficients should continue improving on copying. If compensation is tied to the chosen
shell, moving far away can cause overcompensation; at k=4, small-displacement first-order
analysis predicts frozen L4 compensation residual roughly three times copying error.
Either outcome restricts applicability; it is not evidence of image/geometry/training gains.

No mandatory improvement gate. Report all groups and the counts where each frozen projection
is worse than copying by more than 1e-12. Hash the v2 plan/results/independent audit and this
script before stress evaluation. CPU only, no model updates or CUDA calls.
