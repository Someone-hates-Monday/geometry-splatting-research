# Bounded real-coefficient appearance-transport diagnostic

2026-10-21. Freeze before target-field outputs. CPU only, no CUDA, optimizer, training,
downloads, image metrics or geometry claims. Data are actual published flowers/room 7000
SH3 coefficients; cameras and displacements in this probe are constructed mathematical
shells, not real captured views or recorded densification events.

Use 64 primitive IDs per scene, sampled uniformly without replacement with numpy PCG64
seed 20261021, separately reset per scene. Save every ID before field evaluation. No visibility,
SH-energy, GT, quality or post-probe filtering. All coefficients and IDs retained.

For displacement direction each positive Cartesian axis and a=delta/R in {.01,.05,.1,.25},
target is parent SH evaluated at normalize(d+a*v). Compare coefficient copying, degree3 L2
projection and degree4 L2 projection around the new center. Projections use fixed product
Gauss-Legendre32 x uniform-phi64 quadrature; validate on distinct GL37 x shifted-phi71.
Repeat projection with GL48 x phi96 as a convergence control. Both projection results must
be retained; disagreement >1e-8 maximum absolute coefficient blocks interpretation of
projection improvement. No tuning quadrature based on quality wins.

Read pinned PlenOctree/TGS polynomial evaluator locally and cross-check basis through
degree4 against independently computed scipy spherical harmonics (max abs<=2e-13).
Check Gram4 normalization <=2e-13. Verify zero displacement recovers the source field
and projection coefficients (max abs<=2e-11). No GPU/renderer recapture required.

Primary outcome: per-primitive RGB RMS of unclamped appearance error over the independent
uniform-shell quadrature. Retain all 1536 primitive/axis/displacement cases and both
projection fits. Secondary: native-style lower-clamped RGB, finite-grid maximum error,
degree4 coefficient energy, coefficient-based Lipschitz upper bound for copying.
The upper bound is a real-arithmetic sufficient condition, with 1e-10+1e-10*bound tolerance
only for this CPU FP64 numerical audit, not a formal interval certificate.

This is exploratory case evidence. Report every scene/axis/a separately, plus distributions
over all presampled primitives. No scene/seed p-values. A degree4 win is an approximation
result with 25 rather than16 coefficients (56.25% more appearance storage), not evidence
of runtime efficiency, physical correctness, image quality or a new method.

Rivals: trained fields may be almost constant (copy already enough); highest-degree energy
may create first-order leakage; added coefficients may improve field fidelity but be
uncompetitive in storage; a single shell is insufficient for real multi-distance cameras.
Regardless of outcomes, preserve all results and do not advance to full training.
