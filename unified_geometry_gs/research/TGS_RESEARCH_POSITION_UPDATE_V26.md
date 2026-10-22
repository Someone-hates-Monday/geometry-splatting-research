# 2026-10-21 targeted prior-work check and next falsifiable questions

This is a targeted research-position update during v26 engineering validation,not a
systematic/exhaustive literature review. Search date/cutoff:2026-10-21. Searches used
general public method terms only;no unpublished project code/data were sent externally.

Queries:GaussianPOP Gaussian splatting pruning exact loss local rendering SSIM;
Gaussian splatting densification counterfactual split local rendering loss SteepGS;
triangle splatting paper subdivision densification local loss. Primary pages/full text
below were checked. Search snippets/aggregator criticism were not treated as evidence.

## Primary evidence and scope

[GaussianPOP v1](https://arxiv.org/html/2602.06830v1),Sections3.1–3.3 and Algorithm1:
its deletion criterion measures squared perturbation relative to the current rendered
image and does not require GT. It already uses local cached computations after a render.
Thus “render once/evaluate locally” is prior art,not a new contribution here. This
paper's ideal blending expressions should not be silently promoted to a proof of our
native FP32 cutoff/ordering behavior.

[SteepGS](https://vita-group.github.io/SteepGS/) and
[CVPR2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_Steepest_Descent_Density_Control_for_Compact_3D_Gaussian_Splatting_CVPR_2025_paper.pdf):
splitting is analyzed through a local optimization expansion and a splitting matrix,
not just one immediate before/after image comparison. Inference for this project:
zero-step edit loss does not by itself establish usefulness after parameter adaptation.
We have not reproduced its optimization or final-quality results.

[SAD-GS author project](https://vcai.mpi-inf.mpg.de/projects/SAD-GS/) and
[arXiv record](https://arxiv.org/abs/2604.28016):structure/frequency-guided splitting
already includes a multiview consistency criterion. “Use multiple views for splitting”
alone is not a novelty claim. The author project's PDF link failed through the current
reader;the detailed algorithm has not been fully compared in this refresh.

[CF-Splat v1](https://arxiv.org/html/2610.22108v1),submitted2026-10-15,
SectionsIII-A–III-B and abstract:fixed known mesh,per-vertex SH,linear pixel model and
regularized global least squares. Thus fixed-geometry SH least-squares fitting alone
is also prior art. It differs from our soft-window compositing/topology-edit oracle;
that difference is a scope distinction,not proof of publishable novelty. Its reported
speed/quality numbers are authors' results and are not our reproduced measurements.

Triangle Splatting institutional PDF fetch failed because the reader rejected its
size. Existing archived source analyses remain available,but this refresh does not
claim to reread the whole original PDF. No publication-status or exhaustive absence
claim follows from these searches. Other pruning/density-control and interaction-aware
methods still need targeted review before a method is positioned as novel.

## Important target distinction

Our derivation for any fixed edit,with delta=X-B and a fixed GT image G,is

    ||X-G||^2-||B-G||^2 = ||delta||^2 + 2(B-G) dot delta.

The first term alone measures render perturbation;the cross term controls whether
GT squared loss improves. It may change the sign. This is elementary algebra,not a
new theorem. v23 also provides actual differences between SSE and DSSIM-containing
objectives. Better target alignment is a motivation to test utility,not evidence of
better trained models or a unique explanation for earlier GGGS failures.

## Next questions,not established claims

1. Do signed,loss-aligned topology-edit scores predict benefit on additional held-out
views under a frozen calibration/test separation and consistent SH convention?
Falsifier:advantages vanish or reverse on those views. Known v23 views are exploratory;
do not recycle them as an untouched generalization test.

2. Can independent candidate scores be added when several edits are made together?
Hypothesis to test:disjoint structural pixel influence regions (expanded by the SSIM
radius for loss outputs) give additive local losses under fixed renderer/list semantics;
overlap may create interactions. Compare every frozen pair,including unfavorable cases,
against actual joint-render references. Do not infer necessity from this sufficient
condition,or treat observed zero RGB differences as structural independence.

3. Does immediate edit screening discard capacity that becomes useful after appearance
adaptation? A later bounded adaptation experiment would need its own explicit scope,
baseline and validation. Current no-training diagnostic results cannot answer it.

The useful deliverable so far is a numerically checked,loss-aligned counterfactual
scoring tool with measured conditional cost. Spatial caching,finite SSIM support,
shared moments and basic error identities are not,by themselves,academic novelty.
Independent views/scenes,joint-edit behavior,policy baselines and final geometric/image
quality remain requirements of the original goal.
