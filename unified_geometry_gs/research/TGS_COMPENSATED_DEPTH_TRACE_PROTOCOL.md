# v8 depth failure trace — freeze before tracing or proposing a correction

2026-10-20. v8 failed its unchanged3e-5 real AOV gate on14 pixels of previously used camera9213.
All6 frames and all14 violations are retained in depth_failure_plan.json; other five frames pass.
RGB, paths, median, all gradient groups and counter twins pass; failure is not waived as a small
relative or ULP error. No performance measurement is permitted.5184FD remain independently useful.

Trace ALL14 frozen pixels; no selection by the effectiveness of a repair. Use the exact native
sorted list/cache and sealed/frozen helpers. First bitwise reproduce guard and oracle RGB,
weighted depth, alpha, median and last/median ordinals at every endpoint. Save full packets,
traces and all counterfactuals. Require identical packets from guard/oracle actual native calls.

Counterfactuals, applied only to locally fast windows; native fallback windows remain unchanged:
0 original v8 guard;1 original oracle;2 float-u + double power/raw product;
3 native double-u rounded tofloat + intrinsic power/float raw;
4 float-u + standard powf/float raw;5 native double-u rounded tofloat + standard powf/float raw;
6 Dot2 high/low pair reconstructed in double + double normalization/power/raw product.
All modes retain skip-current stop/cap/cutoff; their own alpha drives their own T. Report any path
change and do not treat a changed-support counterfactual as isolating only a smooth alpha effect.
Pair reconstruction is a diagnostic, not a new efficient FP32 implementation.

Compile trace without use_fast_math, ftz=false, explicit original RN operations and explicit
__powf retained. Thus named powf is the standard function in counterfactual4/5. Bitwise endpoint
checks are mandatory to establish the changed compilation environment does reproduce0/1.
No interpretation if these fail. This is not a power error-bound proof. CUDA12.8 libdevice
documentation distinguishes __nv_powf from __nv_fast_powf; no direct undocumented C++ libdevice
symbol call is used. The existing2e-6 empirical allowance is not promoted to a strict interval.

Independent CPU analysis, using exact Fraction arithmetic on the recorded floats, splits
Dguard-Doracle (only if support identical) into:
1 exact real compositing difference using recorded float alphas;
2 FP32 T/subtraction/blending-weight rounding contribution;
3 weighted-depth FMA accumulation residual difference.
Their exact sum must equal the observed endpoint difference. Also audit RN recurrence/weights.
Compare counterfactual endpoints descriptively, not as proof of an improved renderer or novelty.
Any proposed new version must keep this failure, old127/FD/6real cases, add a fresh holdout,
and pass complete image/path/VJP gates before timing. No optimizer/GT/training.

Source verification: focused web query2026-10-20 'site.docs.nvidia.com cuda libdevice __nv_powf
float fast_powf'; primary archived12.8 page
https://docs.nvidia.com/cuda/archive/12.8.0/libdevice-users-guide/__nv_powf.html .
The current unversioned URL initially returned404. Local header search did not expose a callable
__nv_powf C++ declaration; no assumption is made that its LLVM libdevice prototype is a public
CUDA C++ function declaration. WSL lacks rg; subsequent read-only check used grep.
