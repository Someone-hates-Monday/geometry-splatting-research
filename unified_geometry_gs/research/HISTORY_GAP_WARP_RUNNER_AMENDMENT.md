# Runner-only v1 failure retained

Warp CUDA compiled and completed all20 alternating full raw-chain timings
and kernel timings; the runner then failed constructing a result dictionary
with gate_pass supplied twice. No numerical verdict had been saved.
The failed manifest/traceback/exact source ZIP and actual compiled native
remain. V2 reuses that exact source-checked binary, fixes dictionary merging,
and saves timing progress plus all special-case arrays. It repeats the same
predeclared inputs/timing law/gates because original observations were lost
before persistence. No CUDA arithmetic, graph, thresholds or sample changed.
