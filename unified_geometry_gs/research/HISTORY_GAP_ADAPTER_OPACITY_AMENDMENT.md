# Original saturated logits: preserve, do not silently repair

V1 passed public/query/CSR bit replay and all probe source/raw/ID mapping,
then its global all-parameters-finite check stopped on original opacity
before any optimizer step. Input-only preflight finds3431/1908/2443
positive-infinity opacity logits in the original scan37/55/63 PLYs,
no opacityNaN or negative infinity and no nonfinite xyz/scale/quaternion.
Original files/hashes and v1 source ZIP/traceback remain unchanged.

V2 explicitly treats only these existing positive-infinity logit positions
as fixed opaque coordinates, sigmoid(+infinity)=1. It does not replace,
clamp, prune, drop or zero any original model entry. Their exact mask and
positive-infinity values must remain unchanged; gradients there must be
exactly zero, with no masking of computed gradients. All other parameters,
all gradients and all Adam states must be finite. Parameter comparisons
separately assert identical infinity masks and compare remaining finite
coordinates, avoiding the meaningless infinity-infinity subtraction.
The original blanket finite-parameter gate failed and is not relabeled.

This is a model-compatibility amendment, not a differentiability theorem
at an infinite logit. The loss's coefficient C1 statement is unchanged;
the opaque coordinate is fixed, with the sigmoid limiting derivative0.
It provides no proof that prior training was healthy or that the source
infinities caused earlier quality failures. RGB controls, optimizer
bindings, shared scatter, numerical gates and all cases stay unchanged.
