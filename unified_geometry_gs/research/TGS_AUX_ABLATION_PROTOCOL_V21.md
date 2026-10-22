# v21 auxiliary-arithmetic ablation / output-matched forward control

2026-10-21. Continue fixed-input CUDA diagnosis only; no training or backward.
Create an isolated generated forward.cu from the pinned original; never modify original
or sealed artifacts. Remove only per-pixel maximum-blend atomic update,depth/normal/
distortion/moment arithmetic and median-depth value assignment. Preserve median ordinal,
RGB expression,alpha/T recurrence,window/cutoff/cap,stopping,key generation,preprocessing,
sort and all API allocations/shapes. Unused auxiliary outputs remain zero. The unchanged
backward may compile for ABI compatibility but is explicitly unsupported and never used.
This control ablates auxiliary arithmetic, not every possible allocation/preprocessing
cost of a dedicated minimal RGB renderer. Save exact source diff and hashes.

Correctness before any timing: all6 old/fresh camera baselines and all48 v19/v20 proposal
cases, including old zero edits,large coverage and moved child appearance. Original
captures reproduce their prior hashes. Control full-image RGB,alpha,last/median ordinal
must literally match original, as must tile lists and all visible geometry cache fields.
Both original-child and control-child versions of the immutable v19 cached query must
match the full original reference literally. No tolerance relaxation or case removal.
Save all case receipts and forward counts; cap204 native/control forwards including
12 baselines and48*(original/control full + original/control4-child).

Only after all gates pass, freeze a separate four-arm balanced timing protocol on the
16 v20 stress cases. Arms: A original full,B control full,C cached original4-child,D
cached control4-child. B/D is the primary output-scope/auxiliary-arithmetic-matched
query comparison; A/B and C/D identify ablation effects. Retain API allocation/caching/
event preparation caveats,setup costs,memory tradeoffs and all slower cases. Do not
interpret this ablation as a novel renderer,training speedup,GT improvement or final
attribution to cache reuse until measured. Whole-scene code/quality goals remain open.
