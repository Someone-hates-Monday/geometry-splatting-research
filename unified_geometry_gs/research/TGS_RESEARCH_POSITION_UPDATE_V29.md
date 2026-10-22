# 2026-10-21: primary-source update after joint-policy transfer failed

Targeted search, not an exhaustive review. Public queries only:
`Gaussian splatting densification splitting appearance optimization counterfactual bilevel loss topology`;
`Gaussian splatting densification appearance Schur complement splitting optimization SteepGS`;
`CF-Splat 2610.22108 fixed mesh spherical harmonics least squares`.
No private source, image or unpublished result was sent to external search.

The [SteepGS author project](https://vita-group.github.io/SteepGS/) describes splitting
as restoring optimization progress near stationary configurations and bases decisions
on a local expansion/splitting matrix. It supports treating immediate unadapted edit
loss as a limited diagnostic, not the whole capacity question. The CVPR PDF fetch
timed out in this refresh; its old reading is not presented as a new full-text review.
We have not reproduced SteepGS or established applicability of its Gaussian formulas
to soft triangle windows.

**Correction to the scope of the earlier CF-Splat comparison.** v26 concentrated on
SectionsIII-A/B. Reading [CF-Splat v1, SectionIII-F and IV](https://arxiv.org/html/2610.22108v1#S3.SS6)
now confirms gradient-guided triangle densification before appearance estimation:
multi-view image gradients guide insertion of an interior vertex/three children,
followed by rerasterization; the resulting mesh's SH appearance is solved by the
regularized LLS system with matrix-free PCG. Thus the prior-art boundary is wider
than just “fixed mesh SH fitting.” Combining subdivision with appearance refitting,
or directional SH regularization, is not alone a new idea. Its z-buffer/vertex model
differs from this project's soft-window alpha compositing and per-primitive SH.
That distinction does not establish novelty, and its reported benchmarks are not
our measurements. Code is stated to be released on acceptance; not reproduced here.

The publisher's indexed abstract for [Opacity-aware density control](https://www.sciencedirect.com/science/article/pii/S016786552600351X)
describes a density-preserving opacity correction integrated with SteepGS. The actual
page returned403, so only abstract-level positioning is available in this refresh;
do not claim a detailed derivation comparison or transfer its Gaussian correction
to TGS. Other search aggregators and speculative hypothesis pages were not evidence.

v29's own evidence falsifies a stronger immediate-edit pilot in these two checkpoints:
shared calibration structural visibility and eight disjoint edits still do not make
the calibration-selected mixed image loss improve on the new heldout views. This
is not proof that all zero-step selection fails or that adaptation will succeed.

Next useful test:separate geometry/topology perturbation from locally adapted
appearance, with the unedited model receiving an equal adaptation budget, and tied
versus independent child DC offsets as a capacity control. First prepare and verify
the native color-response coefficients without fitting. Any actual optimizer or
coefficient fitting remains outside the human's present math/small-forward scope.
The two-view SH rank/projection supplement is standard linear algebra and a design
warning, not a new method. Novelty needs deeper prior-art comparison and independent
quality/geometry/efficiency evidence; the original research objective is unachieved.
