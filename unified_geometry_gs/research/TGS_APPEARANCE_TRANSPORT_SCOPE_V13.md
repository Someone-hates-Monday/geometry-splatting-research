# Appearance transport research scope v13

2026-10-21. Narrative targeted review and theoretical method development, not a systematic
review, preregistration registry, or a novelty certificate. Single agent screening. Preserve
v12 and every earlier negative result. No training/optimizer authorized in this phase.

Question: when a splatting primitive's geometry changes (relocation/splitting), how can its
view-dependent appearance be transported or approximated with an explicit error guarantee,
without merely renaming known anchor/shared-feature schemes?

Scope of evidence: official CVF/publisher proceedings and arXiv full reports, author project
pages and official repositories. General web search is supplementary discovery; inspect full
primary text for claims. Search Hugging Face paper metadata as complementary discovery where
available. Include direct prior art on SH evaluation centers, appearance anchors, densification
function preservation, Gaussian splitting and continuous optimization. Exclude unrelated
generative/video work unless it directly bears on the method; preserve relevant negative/overlap
evidence. Search date is 2026-10-21, service indexing may lag actual date.

Initial search families: Scaffold-GS camera-center/view-dependent appearance anchors;
Gaussian splatting densification appearance/color preservation; SH direction relocation and
finite-degree translation approximation; moment-preserving Gaussian/triangle subdivision.
Queries use only public method names/concepts, no unpublished code/data uploads.

Candidate boundaries (not accepted hypotheses): A separate appearance center can exactly
retain the color component of a relocated primitive, but that alone is likely known and does
not retain its coverage/opacity/composition. A finite-degree SH approximation may admit an
explicit displacement/distance error bound. Such a bound can guide subdivision or coefficient
transport only if tighter/useful on actual inputs and distinct from existing work.

Rivals: (1) all useful behavior already covered by anchor methods; (2) geometry/opacity changes
dominate any appearance drift; (3) added SH/anchor memory or evaluation costs erase benefit;
(4) discrete visibility/composition invalidates an unqualified whole-image bound; (5) the
legacy source anomaly is confined to that implementation and has no broader method value.

Next experiment, if motivated by review, must freeze a new bounded protocol before evaluating
outcomes. Mathematical checks can proceed without GPU or full training. Do not infer training
benefit from a zero-step appearance approximation.
