# Max-edge interval tightening after v7 (mathematical supplement, not an implemented speed gain)

2026-10-20. v7 uses Emax=max(e0,e1,e2) for phi=max(r0,r1,r2), even after establishing
which edge attains the maximum. This is valid but can overestimate the relevant uncertainty.

Let stored/native target edges d_i satisfy |d_i-r_i|<=e_i, e_i>=0. In exact arithmetic set
l_i=r_i-e_i, u_i=r_i+e_i, L=max_i l_i, U=max_i u_i. Then

    max_i d_i is in [L,U].

Proof: each d_i<=u_i<=U. Choose an index attaining L; its d_i>=l_i=L, so max_i d_i>=L.
Consequently U<0 certifies inside and L>0 certifies outside. These checks may be more informative
than comparing phi_hat=max(r_i) with max(e_i); the old interval is a containing enclosure.

If for some a, l_a>max_{j!=a}u_j, then for every admissible d,
d_a>=l_a>u_j>=d_j. Thus the active edge is a, uniquely. The same inequalities give r_a>r_j,
L=l_a and U=u_a. It follows that

    |max_i d_i-r_a| <= e_a,

so using max(e_i) after this separation is unnecessary. This is an application of interval
arithmetic and max monotonicity; no claim that the general theorem is new.
The normalization sensitivity bound .99*sigma*du/(uhat-du) should use the corresponding
e_a after strict separation, including the existing native64 error term, not an inactive edge's
larger final-rounding error. Coefficient/projection/power and compositing qualifications persist.

Concrete exact counterexample to equivalence of the OLD and tighter tests:
r=(-2^-20,-512,-256), e=(2^-40,2^-15,2^-16). Edge0 intervals are strictly above other edges;
U=-2^-20+2^-40<0, so inside and unique are certified. The old Emax=2^-15 exceeds |phi_hat|,
so the old sign test is inconclusive. Emax/e0=2^25. All three centers can be exact admissible
targets. This is an arithmetic example, not a measured rendering failure or gain.

Implementation obligations for a later isolated candidate:
- Compute l_i with __fsub_rd and u_i with __fadd_ru. Reject sign using actual outward endpoints.
- Preserve first-max convention when intervals overlap; strict separation is sufficient, not
  necessary. Fall back when required rather than assign a spurious unique derivative.
- Once separation holds, use the active edge's valid original error radius; retaining the old
  max radius remains a valid conservative option. Outward interval radii computed from L/U may
  include additional rounding and must not be tightened by ordinary RN subtraction.
- For a cap/cutoff distance certificate, use an outward interval or a downward-rounded positive
  distance, not an RN absolute difference that may round upward at the margin.
- Propagate uncertainty through normalized u, power and compositing; this theorem proves none
  of the remaining stages automatically. v7 conditional power allowance stays empirical.

`check_tgs_active_edge_interval.py` checks the exact counterexample and the 2560 frozen v7
reached pairs. Only inside-unique samples have all three compensated intervals recorded.
They are one predetermined ordinal per sampled ray, not a uniform draw of all pair evaluations.
Their ratio summaries cannot estimate frame-wide high-precision fallback reduction or speed.
The five v7 cameras are now exploratory inputs for any implementation designed using this note;
a subsequent evaluation must add a previously unused camera or scene and retain all old gates.
