# v21 four-arm output-matched attribution timing

2026-10-21, after six baselines and all48 edits pass literal gates. Use all16 v20
coverage/appearance stress cases without rerouting or selection. No training/backward.

Arms: A_full_aux original full changed scene; B_full_rgb auxiliary-arithmetic-ablated
full changed scene; C_local_auxchild cached query with original native4-child forward;
D_local_rgbchild cached query with ablated4-child forward. Caches use the corresponding
original/control original-scene forward. The primary ratio is B/D, not A/C. A/B estimates
full-scene auxiliary arithmetic ablation; C/D estimates it within the child-query pipeline.

Two unmeasured four-arm warmup blocks/case, then8 measured blocks. Each measured block
contains every arm once. Use a randomized relabeling/shuffle of a four-treatment Williams
design repeated twice, so each arm appears twice at each position and each ordered
within-block neighbor pair appears twice. Freeze orders before timing. No dropped block,
added repetition, outcome-driven stopping or implementation change. Cap644 forwards:
4 original/control cache setups plus16*(2+8)*4 endpoints.

Prepared event tensors and all H/M colors outside endpoints for both full/local methods.
Each local arm includes a newly executed4-child forward of its backend, unpacking,
full output clones and replay; no free reused child cache. Reuse unchanged v19 measure()
for synchronization,wall clock,CUDA bracket and peak allocated-memory increment above
current retained context. Release each endpoint output before measuring another arm.
The first warmup again checks the four selected-output images globally.

Record setup/storage for both cache types, keep all raw8 blocks/case and optional device
telemetry. Summarize per-case median within-block ratios, coverage/scene/appearance
subgroups, slower cases, and API allocation caveats. Setup-inclusive8-query/view numbers
are estimates from sums of case medians plus observed one-time setup, not actual batch
timings. Cold setup comparisons are not randomized causal effect estimates.

Control retains the native API and allocations including unused auxiliary buffers and
unchanged preprocessing. It is an auxiliary-arithmetic-matched control, not the fastest
possible dedicated RGB-only full renderer. Cache reuse attribution remains conditional
on this implementation. No novelty,model-quality,training or general rendering speed claim.
