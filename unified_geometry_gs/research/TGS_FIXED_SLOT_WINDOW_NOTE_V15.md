# Conditional fixed-slot alpha replacement identity

2026-10-21. This is a standard alpha-compositing identity used as a diagnostic, not a new
general rendering theorem or a proof that native TGS satisfies its premise after splitting.

For one channel, write a scene as prefix accumulated color F, prefix transmittance T,
an event block with constant event color c and composite opacity a, and suffix color B
including the background. Then

    C = F + T*(a*c + (1-a)*B).

If F,T,B,c are unchanged and the event block remains contiguous in the ray order,
replacing a=P by a=Q gives exactly

    C_Q-C_P=T*(c-B)*(Q-P).

If all colors and background lie in [0,Cmax] and all alpha lie in [0,1], then B is a
nonnegative convex/subconvex combination of those colors, so |c-B|<=Cmax and 0<=T<=1.
Thus |C_Q-C_P|<=Cmax*|Q-P|. For constant-color overlapping children, Q is their composite
opacity 1-product(1-alpha_j), rather than a plain sum; the identity still holds provided
other primitives do not interleave and prefix/suffix do not change.

If P=Q but ordering crosses an external layer, the bound need not hold. For example take
an event with a=1/2,c=1, one external layer with b=1/2,color0, black background. Event
before external yields color1/2; after external yields color1/4. Composite event opacity
is unchanged, yet RGB changes by1/4. This is an elementary counterexample to applying the
bound without its ordering premise, not a claim about measured v15 cases.

Native FP32 accumulation, alpha cutoffs, skip-current transmittance stop, culling and
tile lists require separate checks. A tolerated empirical inequality is not a rigorous
floating-point certificate. Also, satisfying a necessary upper bound is not proof that
an assumed fixed-slot mechanism actually generated the result. Window sums, pixel errors,
scene quality and optimization utility are distinct quantities.
