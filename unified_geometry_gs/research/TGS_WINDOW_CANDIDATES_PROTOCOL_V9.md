# v9 frozen comparison: single-pass coarse exact vs compensated standard-power

2026-10-20. Mathematics and bounded CUDA only, no optimizer/training/GT. Scientific goal unchanged;
correctness, efficiency, training quality and novelty remain distinct requirements. v8 remains FAILED.

## Two independent candidates, predeclared gates

coarse: source clone of sealed v5, no projection/culling/original-normal/Jacobian/SH/compositing
changes. Precompute9hi plane floats, uniform footprint bound,2extents,domain,reserve=14floats.
For each visited pair, explicit RN32 coarse max above its positive bound certifies native64 outside;
otherwise execute exactly the old native64 edge/normalization/power/raw/cap window. Outside native
windows cannot contribute to any backward loss either. All surviving alphas/active edges remain
native64. Remove the WHOLE preflight invocation; reset existing replay buffer to0 and render once.
No uncertain-alpha/fused provisional contributions, no maxblend rollback, no bounded local list.
This targets measured dominant outside work and double traversal, not a claim that full FP64 is new.
Original double cache/counters retained. Old10float error allocation replaced with14floats, +16B/tri
before alignment, not v8's +152B/tri. Record actual buffers and peak later.

For one edge coefficients c and h=RN32(c), local pixel range bound B=(Bx,By,1), let
D=sum|c-h|B,M=sum|h|B,M64=sum|c|B. E=D+2^-22*M+2^-51*M64 bounds the RN32 ordered plane
against the specified ordered native64 plane under the v6 sufficient domain. Cache max edge E
with all positive operations/conversions upward. Domain: finite integer anchor |anchor|<2^23,
W/H<=8192, local integer coords<=8192, nonzero c/h in[2^-60,2^20]. Invalid means native fallback.
No inverse-radius or power approximation occurs in this candidate. Conditional reference is stored
binary64 planes, not a universal original-geometry projection/tile enclosure. Max monotonicity
proves coarse_max>E implies some native64 plane>0. Native fallback retains native tie/cap/stop.

standard: exact source clone of FAILED v8 except an explicit experimental extern C device
__nv_powf declaration/call replaces the ONE fast __powf in the compensated window. Preserve all
other arithmetic/48float cache/dual traversal/T guard/error budget/cutoff/cap behavior. Do not
disable global fast_math and thereby change other preprocessing/Jacobian arithmetic. Low-level
dispatch verified on CUDA12.8/sm120 finite inputs, not a portable public C++ header guarantee.
2e-6 power allowance remains empirical, FTZ boundary limitations persist. No strict-power theorem.

Compile only new coarse_diag/coarse/standard_diag/standard. Read-only old v5 oracle_diag/oracle
are references (existing binaries with10float errors). No new oracle cache/layout overhead added
to the primary reference. Layout readers dispatch14/48/10 by variant, never infer from values.
All counters disabled in lean clones; unchanged buffers/zeroing in each clone family.

## Inputs and tests (no tolerance adaptation after outputs)

Keep ALL127 v8 inputs byte/hash/reference identical; six variants full images and retained selected
VJPs. Add no invented training data. Same independent barycentric/complex-step checks, FD
directions and step sizes. All2592 step records per lean variant(coarse/standard/oracle),7776 total;
these are reused directions, not7776 independent scientific samples. FD support stability and numeric
error distribution reported separately; never call all FD1e-3 based on support stability.
Prior CPU/native median-boundary disagreement is retained as in v8.

coarse forward MUST be BITWISE native oracle for RGB/all7aux/maxblend, radii/last/median IDs;
all replay flags/preflight counts zero, for every old fixture and real camera. Candidate VJP finite,
relative1e-5 vs native (or inherited1e-6 absolute tiny-reference rule), no bitwise atomic-gradient claim.
Independent CPU physical selected gates unchanged1e-3. Counter twins bitwise all forward/cache/path
and selected VJP agreement1e-5. Single-pass is verified by source/zero preflight/runtime counters,
not by an assumption that dead template kernels cannot appear in SASS.

standard: inherited v8 RGB/aux/maxblend abs3e-5, identical paths/radii/median, flagged endpoints
bitwise oracle, all6VJP arrays finite and relative1e-3; counter twins same1e-5 selected parity.
Keep all14 frozen9213 depth violations explicitly as regressions; their candidate outputs must
meet original3e-5 and paths unchanged. No waiver based on ULP/relative error or new holdout passing.

Real seven FULL cameras: six inherited9258/9322/9269/9221/9213/9326 plus unused test camera
minimum SHA256('tgs_window_candidates_v9/'+name), excluding all six, chosen before any output.
Same stump7000/784056tri/SH3/1245x825. New camera is method holdout, not another scene/seed.
RGB and weighted-depth fixed upstream VJP. Compare against read-only v5 oracle and original
v8 failure-pixel values; record actual flag/path and finite groups. No GT matching or quality claim.

Each candidate has its OWN precision decision. A failure of standard does not block unrelated
coarse audits, or vice versa. The all-candidate aggregate stays false if any failed; performance
measurement is allowed only for individually fully audited passing candidates. Never time a failed
candidate then reinterpret an exploratory failure as success. Baseline/counter/source/hash gates
are mandatory for either. After precision passes, freeze a separate balanced cost protocol against
old native oracle; contextual historical piecewise has different numerical forward and is labeled.
No performance outcome promised. Old v5 negative cost and v8 failures remain preserved.
