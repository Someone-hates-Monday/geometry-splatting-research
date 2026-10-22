# v27 all-pairs joint-edit diagnostic

2026-10-21. Continue authorized mathematical/small CUDA diagnostics. No training,
backward,optimizer,accepted edit,new scene/download or policy-dependent case selection.
All single-edit outcomes are already known;all new joint outcomes remain unmeasured
at protocol freeze. This is a new contrast on existing models/views,not independent
generalization evidence or formal preregistration.

For each flowers/room model retain four v20 parent IDs and their moved-color edits.
Inherited SH coefficients are reevaluated at each known fresh/old_0 camera. All six
unordered distinct-parent pairs in each view are mandatory:24 joint cases. Delete both
original parents simultaneously,retain original relative order,append the first pair
member's four children then the second's. No parameter fitting/scene mutation accepted.

Before rendering freeze source,models/cameras/GT,prior single RGB/metric references,
and all pair classifications from known structural rectangles. R disjoint predicts
pointwise loss additivity;A=R+5 disjoint predicts SSIM/DSSIM additivity. Empty R has
empty A. Overlap is not a prediction of necessarily nonzero interaction. Never replace
structural R with the set of pixels observed to change in a single render.

One original control forward and one16-child preprocessing bank per view plus six
complete joint references:4*(1+1+6)=32 native/control forwards. Reuse unchanged v21
RGB control and v26 shared FP64 loss module. Before each pair verify original field
remapping and both children's visible fields against their independent preprocessing.
Reproduce original RGB/single full-image hashes;verify rectangles against frozen plan.

For each actual joint render,check RGB unchanged outside union bounding region and
save a lossless RGB patch/full hash. For R-disjoint cases require literal global RGB
mosaic and exact signed integer PNG8 SSE additivity. For A-disjoint cases additionally
require literal global GPU SSIM mosaic. FP64 contrast zero tests use the explicit
term-sum bound in the math note;no tolerance can rescue failed literal image/map gates.
Use full-frame loss calls,not the unimplemented joint local replay. Compare single GPU
scores literally to prior full-frame v26 scores. Record every pair's signed raw MSE,L1,
SSIM,PNG8 SSE and L1/L2+DSSIM interaction,not just positive candidates. Retain counts of
nontrivial disjoint pairs (both single RGB effects nonzero) separately.

Software fixtures:simultaneous parent removal/append order;disjoint R and disjoint
SSIM halos;disjoint R with overlapping SSIM halos;overlapping pointwise changes;exact
rational hidden interaction with zero observed single effects. Label all synthetic.

Independent CPU audit reconstructs all joint full images from saved patches/hashes,
recomputes raw float64 and PNG8 losses,compares to unchanged CPU/gpu tolerances
(1e-13+1e-9*abs(reference)),and verifies structural predictions. It does not rerun CUDA
or claim global FP32 compiler proof. Persist each pair/forward count and every failure;
stop on invalid gates,do not silently retry/select cases after seeing outcomes.

No timing or learned-model quality claim is made. Subsequent held-out-view/independent
scene policy tests and parameter-adaptation studies remain outside this diagnostic.
