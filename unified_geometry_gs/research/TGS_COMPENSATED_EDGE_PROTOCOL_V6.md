# v6 edge-only arithmetic probe — frozen before CUDA execution

Date: 2026-10-20. Authorization: mathematics and bounded CUDA diagnostics, no optimizer/training.
This is not a renderer, speed measurement, quality evaluation, or novelty claim. v5 failed its
overall efficiency objective; its sealed evidence remains unchanged.

## Reference and conditional derivation

For one stored binary64 plane let P=cx*x+cy*y+cb denote exact real arithmetic on these
stored coefficients and integer local coordinates. This does not remove projection/cache error
relative to original world geometry. Split each c as h=RN32(c), l=RN32(c-h), residual delta=c-h-l.
Let A=(hx,hy,hb,lx,ly,lb), Q=(x,y,1,x,y,1), D=A dot Q, S=sum(abs(Ai*Qi)).
Use the standard Dot2 algorithm with TwoProductFMA and the unrestricted TwoSum, all RN32.
Ogita, Rump, Oishi (2005), Algorithm 5.3 and Proposition 5.5, give, without underflow/overflow,
|r-D| <= u*|D| + gamma_6^2*S, u=2^-24, gamma_6=6u/(1-6u).
The subscript is 6, not 12; the exponent is a square. No claim of a new Dot2 algorithm.

With B=8192, M6=sum((abs(h)+abs(l))*[B,B,1]),
Delta=sum(abs(delta)*[B,B,1]), we obtain the a posteriori enclosure
|r-P| <= (u*abs(r)+gamma_6^2*M6+Delta)/(1-u).
It is conservative to replace gamma_6^2 by 49*u^2 and 1/(1-u) by 1+2u.
All positive arithmetic constructing bounds is rounded upward, including cached M6 and Delta.
The independent audit computes exact rational errors and caches; it does not compare only to FP64.

Sufficient domain gate, checked before using any certificate:
- x,y are integers, abs(x),abs(y)<=8192;
- each original coefficient and each nonzero h/l is in [2^-60,2^20] in absolute value;
- all values are finite.
Under this deliberately restrictive gate, split inputs are multiples of 2^-83 and Q is integer;
every exact product and EFT/addition intermediate lies on that lattice. Nonzero intermediates
cannot underflow (2^-83 >> 2^-126); upper magnitudes stay far below overflow. This also permits
the same certificate under FTZ=true within the gate. A tiny coefficient must NOT silently become
h=l=0 and pass: the gate tests original coefficients too. Outside the gate always means fallback.

To compare with the ordered v5 binary64 plane evaluation, add 2^-51*M64,
M64=sum(abs(c)*[B,B,1]), an upward bound for gamma_3(binary64)*M64 in this domain.
For a cheap coarse rejection compute f=RN32(RN32(RN32(hx*x)+RN32(hy*y))+hb),
Efast=DeltaHi+2^-22*M6, DeltaHi=sum(abs(c-h)*[B,B,1]); add the same binary64 term
when targeting the native oracle. f>Efast certifies an outside plane. A max of three
planes inherits max individual error; a triangle is outside if any plane is certified positive.
This probe evaluates individual planes; it does not measure actual ray rejection fractions.

## Frozen input selection and outputs

Use ALL visible stored planes in all 127 v5 retained cases. Also select 512 visible triangles
without replacement from EACH of the four saved v5 real camera caches, seed 202610208.
These cameras were already used in v5 and are NOT new holdouts. For each plane generate:
one integer point in [-1600,1600]^2, one in [-8192,8192]^2, and two integer nearest-line
points with adjacent y (clamped to the domain); retain label of this artificial sampling.
These are coefficient probes, NOT actual accepted/rendered primitive-pixel distribution.
Add fixed cancellation and out-of-domain FTZ controls before either GPU executable runs.
Freeze NPZ, binary input, metadata, source hashes and protocol hash first; do not adapt selection
or tolerances after native outputs. Keep all failures. No outlier deletion, timing, or tuning.

Compile the same standalone kernel twice, CUDA12.8 sm120:
strict: -O3 --ftz=false --prec-div=true --prec-sqrt=true;
fast_ftz: -O3 --use_fast_math --ftz=true.
Use explicit __fadd_rn/__fsub_rn/__fmul_rn/__fmaf_rn and upward bound intrinsics.
Save full EFT traces, ordered binary64 plane, gates, compiler logs, binaries and SASS.
All in-domain rows must satisfy exact rational r/P and r/native64 envelopes; all coarse or
compensated certified-positive rows must be truly positive against their stated reference.
All in-domain products and TwoSum decompositions must be error-free; replay correction RN
for a frozen first 1024 in-domain rows and every control. Rejected-domain controls are reported,
never counted as certified successes or representative real rays.

Power grid: freeze positive float32 u on log scales down to subnormal and sigma in
{.02,.2,.216805,1,3.532998,10}; compare named powf, explicit __powf and binary64 pow.
Under fast_math the named powf is also intrinsic. CPU mpmath at 100 decimal digits is an
independent numerical reference, not an interval proof. Report errors and all controls;
no accuracy/quality promotion based on this finite grid.

## Primary-source scope

Focused primary-source verification, not systematic review or novelty assessment.
- Ogita/Rump/Oishi, DOI10.1137/030601818, author PDF https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf
  Algorithms3.1,3.5,5.3; Proposition5.5. Online text and algorithm sections inspected.
- CUDA12.8 guide https://docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide/index.html#mathematical-functions-appendix
  Sec13.1, Tables15/18, explicit intrinsics and FTZ conditions. IMPORTANT: tabulated power
  errors are based on extensive but non-exhaustive tests, explicitly NOT guaranteed bounds.
  Thus 4 ULP for standard powf is not a rigorous power enclosure; fast_math replaces it by __powf.
- Shewchuk https://www.cs.cmu.edu/~quake/robust.html : adaptive filtered predicates are established
  prior art. Landing-page scope only; the long paper has NOT been full-text assessed in this phase.

Queries (web search, 2026-10-20): Ogita Rump Oishi Accurate Sum and Dot Product 2005 Dot2 pdf;
site.docs.nvidia.com cuda 12.8 floating point fused multiply add __fmaf_rn;
Shewchuk adaptive precision floating point robust geometric predicates 1997 paper cmu;
site.docs.nvidia.com/cuda/archive/12.8.0/cuda-c-programming-guide powf ulp error 2;
site.cs.cmu.edu/~quake robust predicates 1997 adaptive precision floating point arithmetic pdf.
The earlier search response lost at context truncation was rerun; no conclusions depend on it.

## Promotion boundary

Passing this probe permits designing a new isolated renderer candidate only. It does NOT
permit performance, GT quality, training, paper-ready novelty, or production promotion.
Renderer integration must retain 127 cases, 5184 FD records and four real camera gates,
add a previously unused holdout, and retain compositing uncertainty/forward-backward agreement.
Power enclosure and two-pass cost remain unresolved. Do not edit the open LaTeX or original library.
