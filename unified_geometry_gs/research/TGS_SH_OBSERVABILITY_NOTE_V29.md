# Two-view independent-primitive SH observability: scope before an adaptation test

This standard linear-algebra supplement informs a possible next test; it is not an
innovation claim or a fitted-model result. No parameters are optimized here.

Fix geometry, opacity, rasterization lists and cameras. In this project's physical
centroid-direction convention, a primitive t has ONE direction d_tv per camera v,
shared by every pixel it covers. Write its degree-L SH vector y_tv in R^q,
q=(L+1)^2, and one-channel coefficients c_t. Preclamp color is y_tv^T c_t+1/2.
Let Y_t stack the K calibration directions, whether or not that primitive actually
contributes to all these views. Then rank(Y_t)<=K and dim ker(Y_t)>=q-K. For every
z in ker(Y_t), c_t+z has identical calibration preclamp colors. ReLU/clamp preserves
their equality, and all calibration renders remain identical in real arithmetic,
because alpha/ordering/cutoff do not depend on RGB. Occlusion only loses information.

For degree3 and K=2, an independently parameterized primitive has at least14 invisible
coefficient directions PER COLOR CHANNEL. More pixels of the same centroid-direction
primitive in those same views do not supply more directional SH samples. This is NOT
a claim that the image Jacobian has rank exactly2; its rank may be lower. Nor does it
apply unchanged to a group sharing coefficients across several distinct centroids.

Let P project orthogonally onto ker(Y_t). At a test direction y_*, any invisible
coefficient perturbation with norm<=epsilon changes the unclamped color by at most
epsilon*||P y_*||, and this maximum is attained by z=epsilon*P y_*/||P y_*|| if nonzero.
The clamp is1-Lipschitz, so the same upper bound holds after clamping. Attainment
after clamping needs an active-sign condition; a nonzero projection alone does not
prove actual image change or quality degradation on a culled/occluded primitive.
This is a sensitivity bound, not a prediction of what a particular optimizer learns.

The CPU supplement evaluates Y on true checkpoint points for the eight selected
parents and their32 midpoint children per scene, using float64 camera metadata and
float64 centroid/direction evaluation. These are ideal-field numerical diagnostics,
not the exact native FP32 directions or proof of bitwise invisible CUDA edits. Saved
SVD tolerances, singular values, projected test directions and residuals expose that
boundary. No GT images or residuals enter the computation. It is post-selection
description, and changes no v29 choice.

An unconstrained degree3 refit from only these two views would therefore be poorly
identified without additional assumptions. A bounded DC-only experiment, tied offsets,
or more calibration directions can test different questions; none is guaranteed to
improve heldout quality. Direction-aware regularization and linear appearance fitting
already have prior art; see the dated primary-source update accompanying this phase.
