# Next work package: prepare a bounded appearance-adaptation comparison

Status: proposal only. Current authorization covers mathematics and small fixed-input
CUDA diagnostics, not optimizer steps/fitting/training. No adaptation is executed.
The requesting user's original restriction is the reason authorization will be needed
before a fitting stage, not a restriction invented by a skill or a tool failure.

## Prepare under the existing authorization

Use the fixed v29 primary eight-parent sets in both scenes, without further topology
selection. Derive and implement native accepted alpha/transmittance color-response
capture at selected primitives for both original geometry and midpoint geometry.
Record all selected contributors, exact cutoff/skip-current behavior and the fixed
external FP32-color convention. No fixed top-K approximation.

Prove the real-arithmetic conditional color-affine relation with geometry, alpha,
sorting and event list fixed; distinguish it from the rounded FP32 implementation.
Validate source-conditioned finite error bounds and explicit fixed positive/negative
DC perturbations against full native references before using a differentiable proxy.
The zero-offset proxy must reproduce its saved native baseline. Preserve failures
and stop if identity/finite-error gates fail. This preparation contains no fitting.

Freeze code, coefficients, calibration source labels, fitting parameters, candidate
sets and proposed heldout camera selection before requesting fitting authorization.
The currently known v28/v29 test views can be development diagnostics only. A future
evaluation must pick new test views before their image payload or model outcomes.

## Concrete fitting stage to request after preparation passes

Local copies only; original checkpoints remain immutable. Freeze geometry, sigma,
opacity, ordering, cameras, high-order SH and every unselected primitive's appearance.
Maximum100 optimizer steps per branch, alternating the same two calibration views,
with fixed Adam settings, no learning-rate/iteration/seed search. A numerical budget
and wall-time cap must be frozen in the actual executable before approval; source
review/gradient checks and the fitted native-reference gate are prerequisites.

Three branches per scene:

1. Original geometry:DC offsets on the eight original parents (24 RGB parameters).
2. Midpoint geometry:one tied DC offset shared by each parent's four children
   (also24 RGB parameters), isolating topology with equal appearance degrees of freedom.
3. Midpoint geometry:independent child DC offsets (96 RGB parameters), exposing the
   extra-appearance-capacity factor separately.

Retain all no-fit original/split controls. Specify preclamp offset semantics and
trust bounds explicitly; avoid silently turning postclamp color offsets into SH
coefficient updates. Use equal optimizer-step schedules and report actual wall time
and allocation per branch, since equal steps are not equal compute or equal capacity.
The low-dimensional proxy must retain the full native loss normalization and SSIM
neighborhood. Final evaluations must be actual full native renders, not proxy values.

Do not optimize independent degree3 SH from two views without a separate identifiable
model/regularization argument. The rank supplement supplies a lower bound on its
unobserved directions, not a guarantee that DC-only updates generalize.

The question is whether midpoint geometry has useful capacity AFTER bounded appearance
adaptation, beyond equally adapted original geometry. Falsifiers include no heldout
advantage, only calibration improvement, or gains explained entirely by the96-parameter
branch. Use new heldout images, MSE/L1/SSIM/PSNR, parameter displacement, time/allocation
and actual clipping behavior. No final geometry or full training performance claim.
If all branches fail, preserve the result rather than increase steps or change views.

This document is not an executable protocol, not permission to run100 steps, and not
a claim of novel optimization. Prepare the concrete diagnostic implementation first;
then ask the human to expand the fitting scope with this reviewable bounded design.
