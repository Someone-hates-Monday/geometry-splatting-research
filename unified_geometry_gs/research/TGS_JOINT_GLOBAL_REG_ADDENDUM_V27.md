# Global regularizers are outside the image-support additivity result

2026-10-21. Source-informed mathematical addendum after the v27 image audit. No training,
new native render or measured full-objective improvement. SymPy checks identities in
exact arithmetic;this is not a reproduction of FP32 training reductions.

Archived source `sources/tgs_subdivision_static_v1/triangle-splatting/train.py` includes
`abs(get_opacity).mean()*lambda_opacity` in addition to image/normal/distortion terms.
It also uses inverse mean triangle area as a size term before its densification cutoff.
The actual historical public checkpoint hyperparameters are not established here.

Suppose effective nonnegative opacities sum to S over N primitives,mu=S/N,and k distinct
parents with opacities o_i are each replaced by four children inheriting that opacity.
Then count increases by3k and opacity sum by3*sum(o_i). For enabled mean-opacity penalty
R=lambda*S/N,

    Delta R_k = 3*lambda*(sum(o_i)-k*mu)/(N+3k).

The sum of k independently evaluated one-edit changes is instead

    sum Delta R_i = 3*lambda*(sum(o_i)-k*mu)/(N+3).

Thus the global interaction,irrespective of spatial separation,is

    eta_R,k = -9*lambda*(k-1)*(sum(o_i)-k*mu)/[(N+3k)(N+3)].

For k=2 this is -9*lambda*(o_i+o_j-2mu)/[(N+6)(N+3)]. For lambda>=0 and opacities
in[0,1],selected-sum minus k*mu is bounded in absolute value by k*(N-k)/N for k<=N:
write it as (N-k)/N times selected sum minus k/N times unselected sum and bound each.
For a pair the resulting tight opacity-domain bound is

    |eta_R,2| <=18*lambda*(N-2)/[N(N+6)(N+3)].

This is a deterministic ratio correction,not a learned rule or lambda search. Actual
mean-opacity change may also matter for a weak image-only candidate,not just the small
pair interaction. A score oracle for the full objective must add the appropriate global
term and other enabled losses;disjoint image supports alone cannot certify it.

General form:for an edit changing opacity sum by dS and primitive count by dN,

    Delta R = lambda*(dS-mu*dN)/(N+dN).

The inverse mean area source term behaves differently. If total area A>0 is exactly
preserved by ideal midpoint subdivision,its value is lambda_size*N/A and k edits add
3k*lambda_size/A exactly,so this term IS additive in that ideal case. Floating midpoint
geometry can perturb total area;do not claim exact FP32 area preservation. If sum area
changes by dA,the direct change is lambda_size*((N+3k)/(A+dA)-N/A),which must be used.

The symbolic check retains an exact rational toy opacity example demonstrating nonzero
global interaction despite arbitrary spatial separation. It is not a real-model numeric
measurement. No full-objective policy or final image/geometry improvement follows.
