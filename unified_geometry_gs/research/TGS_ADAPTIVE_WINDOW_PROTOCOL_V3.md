# Adaptive edge precision v3 (frozen before native outputs)

2026-10-20. Scope: mathematical verification and fixed-input CUDA, no optimizer,
training, model change, sigma clamp, performance timing or novelty claim.
Retain the immutable v2 79 inputs and all its failures. Add 16 boundary inputs:
sigma .01/.02/.1/.5, widths256/8192, both windings, z2 dyadic world vertices,
near a 3-4-5 hypotenuse. Two fixed tangent pixels plus two interior pixels,
four signed VJPs with seed202610204. No native-output-based selection.

Three isolated direct-Gamma modules: (1) double pixel arithmetic on existing
float coefficients, (2) adaptive fallback to coefficients built in double from
original projected points, (3) always double coefficient/window oracle.
The arithmetic-only module is a causal ablation, not required to pass original
geometry accuracy. Adaptive and oracle must pass the retained selected RGB/aux
absolute3e-5, geometry/scalar/SH VJP relative1e-3 (zero group abs1e-6), old tie
null-direction abs1e-6, and preserve all raw results, including failures.

Compute every pixel of all 95 images against independent CPU64, for RGB and
the original 18-case complete-image reference; require max RGB abs3e-5 for
adaptive/oracle. Report all outliers and auxiliary discrepancies, no discarded
boundary pixels or silently changed branch tolerance. Compare adaptive against
oracle images, but do not require bitwise identity across different arithmetic.
Reuse all 5184 old v2 FD step records as fixed directions/steps if VJP gates
pass; inspect native tile membership and alpha support. FD numerical precision
has no acceptance threshold and must be reported separately. New near-edge
geometry perturbations may have no representable smooth FD; do not pretend
they do. All near-edge VJPs still compare against CPU64/complex step.

Adaptive dispatch uses outward-rounded bounds for the difference between the
float edge/pixel computation and the stored double coefficients. From
|d alpha / d log u| <= .99*sigma, if uhat>delta_u,
|delta_alpha| <= .99*sigma*delta_u/(uhat-delta_u).
Fallback when sign/positive-u is uncertain or this bound exceeds1e-6;
also fallback within bound+2e-6 of contribution cutoff/cap.
The 2e-6 allowance for __powf is empirical, not a proved libdevice enclosure.
Double projection and coefficient construction are a numerical reference,
not certified exact real arithmetic. Do not describe this as a universal IEEE
certificate or the heuristic one-pixel tile guard as a proved coverage bound.

Precompute double normals, offsets and inverse radius (10 doubles/triangle),
plus 10 outward float cache-error bounds and three uint counters. These are
diagnostic memory/counter costs and are included in receipts, not hidden.
Counters count window evaluations/high precision/sign uncertainty in forward
only; they are not unique pixels or nonzero-loss-gradient counts. Never time
this instrumented module. Backward calls the same window helper and retains
direct Gamma; float geometry-chain approximation is tested, not assumed exact.

Any new failure defers promotion and keeps performance disabled. An independent
audit of arrays, formulas, input/source hashes and expanded image scope is
required before a later explicitly versioned efficiency phase.
