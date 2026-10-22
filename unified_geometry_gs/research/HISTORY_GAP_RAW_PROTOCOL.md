# Squared history gap: raw chain and local descent protocol

Fixed before results, 2026-10-15. Unit: each of the existing 891 complete
development rays / 51524 nodes, retaining nine previous anomalies. This is
a continuous local extension through the captured inputs, with fixed
camera/filter/visibility/IDs, as documented by WORLD_ROOT_VJP_RESULT.md.
It is not the literal IEEE program derivative or a shared-parameter scene
optimization. No production code/model mutation or training in this stage.

Compare independently derived MP world inverse + analytic coefficient
adjoint chained through filtered scales, normalized raw quaternion, raw
opacity, amplitude compensation and active alpha cap to PyTorch factor
and dense covariance automatic differentiation. Both Torch branches solve
their own roots from source-derived brackets, without an MP root input.
Use MP geometry140/root and chain240 digits. Gates: whole-ray relative
L2<=1e-5; xyz/log-scale/quaternion/opacity blocks each<=1e-4. All891 must
pass for the raw-chain gate. Preserve candidate/public center contrasts
from the prior stage rather than interpreting them as equivalent roots.

Finite differences: global ray indices given by unique integer linspace
0..890 with12 entries, union all nine old anomalies; random seed13017.
Two fixed directions per selected ray: random and negative gradient,
scaled in coordinates S (xyz=min captured filtered sigma per primitive,
log-scale=1, quaternion=raw quaternion norm, opacity-logit=1).
Normalize directions to ||S^-1 d||2=1. Recompute all fields and solve a
new root at +/-h for h=1e-4,3e-5,1e-5. Last-step error gate:
abs(FD-pred)<=max(1e-9,1e-5*abs(pred)). Report all three steps; signals
below1e-10 are marked numerically weak even if absolute tolerance passes.
Exact cap/screen transitions, missing roots, numerical failures and
zero-gradient directions are retained as distinct outcomes. Add the
previous explicit synthetic3-layer cases with screen kernel0/.3/1 because
all real kernels are zero; do not mix them into real-data success counts.

For all891 rays, test ONE negative scaled-gradient direction at fixed
steps1e-4,1e-3,1e-2 (no adaptive line search). Compare independently
recomputed root-composed losses. Record absolute/relative change and
Armijo Lnew<=Lold-1e-4*h*||Sg||. Exactly zero-gradient directions are
unchanged, not successes. Finite-root loss and unchanged/cap/absent-root
cases remain in denominators. Count high loss L>=.25 and weak scaled
gradient ||Sg||<=1e-8 separately, along with complete distributions.
Also report decreases larger than max(1e-12,1e-8*Lold), to distinguish
visible local progress from an Armijo inequality rounded at zero change.
These diagnostic thresholds do not establish a universal utility scale.

Local descent is only a necessary diagnostic. Rays share primitives, so
independent per-ray steps cannot be claimed as a realizable joint update.
Before CUDA/training adoption, at least a shared-ID aggregate direction
must be checked on actual repeated primitives, and high-loss weak-gradient
coverage and finite-root failures assessed. No photometric/GT improvement,
whole-training speed, novelty, or population failure-rate claim follows.
