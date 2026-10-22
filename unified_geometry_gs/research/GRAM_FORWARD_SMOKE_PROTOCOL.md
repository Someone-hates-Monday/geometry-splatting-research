# Gram determinant forward: isolated full-backend smoke only

2026-10-12, after the independently recorded 164-case forward audit and before
observing this full-backend smoke. This does not replace the active factor64
quality candidate or modify `FACTOR_PREPROCESS_PROTOCOL.md`.

With M the 3x2 projected scaled rotation factor, det(M^T M)=||m1 cross m2||^2.
For image kernel k>=0, det(M^T M+kI) adds k trace(M^T M)+k^2. These standard
Cauchy-Binet/Lagrange identities remove the final subtraction of two nearly
equal Gram products, but do not guarantee uniform conditioning or avoidance of
underflow. Fixed saved inputs all pass the existing 1e-4 forward scaled gate;
all four previous conic failures are retained as control observations.

Build a new full backend whose backward is byte-identical to the audited
factor64 backward, and whose local forward function is byte-identical to the
Gram probe. The production call uses the returned determinant when computing
conic. No bounding, root solver, query grouping, NCC or loss schedule is changed.
The null-scale covariance fallback is outside the certified scope.

Run 20 actual Adam updates from the same verified iteration-6999 state. Require
finite loss and participating gradients, exact same checkpoint/sidecar/initial
state/first camera/initial Gaussian count. Since forward has changed, require
each first forward scalar loss to differ from the completed factor64 smoke by
at most 1e-4 * max(abs(reference),1e-5), rather than requiring bitwise equality.
This is an engineering smoke gate chosen before the smoke data; it is not a
quality threshold or an adjustment of any prior failed gate.

This script never starts full Gram training or claims image/geometry benefits.
A separate justified quality protocol is required after the existing factor64
pilot result, with fixed controls and all failures retained.

Workflow reference: Kassis et al. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
