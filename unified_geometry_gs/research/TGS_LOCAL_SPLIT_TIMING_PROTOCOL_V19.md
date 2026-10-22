# v19 timing frozen after all32 correctness gates

2026-10-21. Cached single-event query latency/memory; not training/inference speedup.
Use all32 same cases, no case dropping. Per-case2 paired warmups, then8 measured pairs,
exactly4 full-first and4 local-first in a deterministic SHA256/PCG permutation.
Cap644 native forwards total:4 original cache setups plus32*(2+8)*2 endpoint forwards.
One local endpoint contains one native4-child forward plus cache unpacking, output
clones, mask and merged affected-tile replay. Its native4-child full-image allocation
and rendering are INCLUDED. Full endpoint is unchanged native whole split rendering.

Prepared event tensors/colors and the baseline cache are outside paired endpoints for
both methods. This isolates trial-render query cost. Do not describe it as complete
densification cost; event definition/full-array construction, baseline setup and cache
retention still matter. Record baseline setup(native forward+unpack+alpha) once/view,
and retained baseline tensor storage plus colors, explicitly separate from model inputs.
The native endpoint produces auxiliary fields not implemented by the RGB/alpha/path query;
this is an evaluation-query comparison, not two interchangeable training renderers.

Synchronize before and after every endpoint. Record perf_counter wall latency with CUDA
event bracket and peak allocated CUDA bytes above that endpoint's starting active bytes.
Keep outputs alive through stop synchronization/peak read, then release before the other
endpoint, so the first endpoint's output does not inflate the second's active memory.
Warmup first pair checks global RGB/alpha/path identity again outside timing. Do not put
CPU copies, hashes or cache/list audit kernels in measured endpoints. No clock locking,
outlier exclusion, added repeats or stopping rule based on speed results.

Report raw8 paired measurements/case, case median paired full/local wall ratio as primary,
and case medians of wall/CUDA bracket and peak-allocation differences. Summarize four
views, both scenes, all cases and affected_tiles>0 /==0 groups fixed by correctness
before timing. No pixel/sample-independent p-values or scene-general confidence claim.
Record hardware/runtime and optional before/after nvidia-smi telemetry, including failure
to read it. Cached cost excludes model loading and one-time original forward; any speed
claim must state this amortization and the fixed-checkpoint single-parent scope.
