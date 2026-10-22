# v2 retained failure and v3 cancellation amendment

V2 completed all 891 rays / 51524 nodes, retaining all nine old anomalies.
All gradients were finite; the maximum whole-ray relative L2 was
5.693847110522264e-12, but only 783/891 passed the unchanged block gates.
The 108 failures expose cancellation in 1-r when r rounds to one, despite
nonzero small peak/rho blocks. Five source nodes had zero-rounded alpha.
The public Float32 center passed only 239/891. V2 artifacts/source ZIP
remain unchanged.

V3 sums positive after-peak density A and positive before-peak density B
separately. Set r=(A-B)/(A+B), and compute the factor 1+r*s directly as
2*A/(A+B) before the peak and 2*B/(A+B) after it. This avoids subtracting
nearly equal values. Both MP and Float64 analytic evaluations use the
identity; the independent root-composed finite differences are unchanged.
All cases, precision, and original pass/fail rules remain unchanged.
No training, complete raw chain, CUDA accuracy or quality claim is added.
