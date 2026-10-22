# v22 four-arm integer-score timing

2026-10-21, after all16 exact score/debug/cache/list gates and CPU reconstruction.
Freeze sizes1,2,4,8 as nested prefixes of each view's unchanged8 proposals. Their
coverage mix changes with prefix size; smaller-size results are diagnostics,not a
causal hardware scaling claim. Primary size8 uses all proposals. No GT-driven selection.

All methods return the SAME K int64 PNG8 SSE differences:
A full-scene control render plus full-image integer delta reduction for each proposal;
B sequential v19 cached RGB-image query plus that delta reduction for each;
C sequential scalar ROI query (fresh4-child control forward,ROI plan/scan/score for each);
D batched scalar ROI query (fresh4K-child control forward,one ROI plan/scan/score bank).
C/D separate batching from eliminating cloned images; B/C separates image-query/reduction
overhead. Native control21 and query19 remain unchanged. All native API allocations and
the bank's unused mixed-image rendering count in endpoints. D may not reuse a child
cache/ROI plan for free. Scalar kernels return score/stat tensors,not full images.

Prepare proposal/event tensors/colors,GT bytes,baseline PNG bytes and static bank slices
outside endpoints for every method. Each size uses the same prepared proposal data.
Record complete original-control baseline forward/cache-unpack/PNG setup once per view,
retained cache+PNG+GT storage and model-input exclusions. Warm context is primary.
Also provide explicit estimates under two contracts: given original PNG but no geometry
cache (charge setup to local),and original render required by both (charge common setup
to both). These are conditional arithmetic estimates,not timed cold batches. Do not
silently compare the new score operation's setup semantics to old image-only queries.

Per size/view:two four-arm warmup blocks then8 measured blocks. Use randomized/relabelled
four-treatment Williams design twice:each arm twice/position and each within-block ordered
neighbor pair twice. Reuse v19 synchronization/wall/CUDA bracket/peak helper; release each
endpoint output before another. First warmup checks all four score vectors against
independent saved CPU references and zero diagnostic flags. No outliers,added repeats,
rerouting or performance-based stopping. Cap954 control forwards:
2 views*[1 baseline+(2+8)*sum_{K=1,2,4,8}(3K+1)].

Retain all blocks and sizes,report within-block case medians,slower samples and raw time.
No clock locks,p-values across pixels,independent-scene confidence,policy/model-quality,
training improvement or broad novelty claim. API allocations/preprocessing are not a
minimal dedicated full-scene score backend. No proposed edit is accepted into the model.
