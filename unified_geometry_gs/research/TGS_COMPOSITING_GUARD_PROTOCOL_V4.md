# Transmittance classification and oracle-ray rendering, v4

Frozen before v4 native outputs,2026-10-20. No training, optimizer, GT quality,
latency timing, epsilon change or include-current semantic change.
The previous turn was substantive progress; v3 remains sealed.

The trace_v1 ablation has reproduced native RGB/depth/alpha/median/contributor
endpoints bitwise at both frozen real pixels, with identical sorted lists and
coefficients. First early-stop disagreements are ordinal338 /760. Preserve this
evidence and every previous failure.

Retain all95 v3 inputs. Add16 compositing stresses using CPU64 geometry-only
opacity construction: tail targets [.9,.9,.9,.9,.3], median [.5,.4,.3], sigma.13,
fixed 3-4-5 screen triangle, positive power-of-two depths, both windings,
opacity ULP shifts -1,0,+1,+4 on the relevant threshold layer. Four signed VJPs
seed202610205, fixed4pixels. All selected before native output, no outcome tuning.

Implementation uses two passes for adaptive: a classification pass propagates
outward transmittance intervals, then an ordinary final render uses the double
window path for flagged rays. Classification never writes max_blending or image
outputs, so it cannot leave spurious contributions from a rejected provisional
path. This duplicates window work and is an explicit prototype cost. Do not
claim sparse second-pass work or a speedup from the replay fraction alone.
Cache flags per pixel for backward. Keep counters for both passes separately;
add12 bytes/triangle and4 bytes/pixel beyond v3, plus alignment.

Classify if incoming-T interval straddles.5, or next-T interval straddles1e-4.
Use the existing adaptive alpha-error bound plus the empirical2e-6 power allowance;
double-window fallback has the same deterministic high path as oracle. Freeze
double edge sums with explicit rounded operations to avoid context-dependent
contraction of that expression. The resulting mathematical window is unchanged,
but arithmetic is explicitly versioned; compare against both v3 and v4 oracle.
This is not a universal interval enclosure for libdevice/projection.

For95 retained cases, require CPU64 selected/full RGB abs<=3e-5 and VJP groups
relative<=1e-3 (zero max abs<=1e-6), old tie-null direction<=1e-6, as before.
For16 new near-compositing-threshold cases, report CPU64 values and branch choices;
the main test is adaptive vs v4 oracle RGB/aux/contributor paths, not equating
float compositing to exact real threshold decisions. Require RGB/aux max3e-5,
identical per-pixel contributor/median IDs, radii identical, and corresponding
VJP groups relative1e-3. Never hide CPU64 threshold disagreements.

Execute all old5184 FD records only if candidate precision gates pass; preserve
tile/alpha branch checks and report numerical FD error separately. Real complete
stump7000 two train cameras use unchanged frozen parameters/resolution, compare
RGB and every auxiliary channel max3e-5, contributor/median IDs identical,
maxblend abs3e-5, all gradients finite; RGB and weighted-depth VJPs compare to
oracle with relative1e-3. Both old outlier pixels must be flagged. Failed gates
defer promotion; keep outputs. Independent audit required, no timing in this phase.
