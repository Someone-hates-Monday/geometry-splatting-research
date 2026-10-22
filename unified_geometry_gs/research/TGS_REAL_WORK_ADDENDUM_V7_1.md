# v7.1 observer sentinel repair, frozen before corrected native execution

2026-10-20. Initial v7 execution completed all5 cameras and ended exit1 at final gate assertion;
it is TERMINAL, not a timed-out/live job. All original inputs, outputs, results.json gate=false
and execution_failure.txt remain unchanged.

Only camera0 path channel1 mismatched627 pixels. Independent NumPy inspection established:
last ordinal agrees everywhere; all627 mismatches coincide EXACTLY with included K==0;
observer value is uint32(0xffffffff), native value uint32(0). RGB, all7aux and maxblend all agree.
The original renderer declares float median_contributor=-1 then assigns it to uint32; observer
incorrectly initialized a uint32 directly to0xffffffff. These are different CUDA conversions.
Other4 cameras have no zero-K pixels and all initial endpoints agree.

Correction generates a NEW source copy with exactly ONE replacement:
`mid=0xffffffff;` -> `mid=__float2uint_rz(-1.0f);`.
No arithmetic, selection, counters, cache, alpha, stopping logic, original library or model change.
The observed native zero value is retained as evidence; never merely overwrite old outputs to pass.

Compile a separately named v7.1 module using the same flags/helper. Re-run all5 frozen cameras
without timing; compare ALL observer buffers to saved v7, with the ONLY allowed difference being
path[1] for old zero-K pixels, which must now match the freshly executed native oracle exactly.
Full native RGB/aux/path/maxblend parity must hold in the corrected calls; full observer counts,
split/cache/B/domain hashes, sampled pairs (including inactive NaNs), RGB/aux/maxblend and all
input/model hashes must match v7. Save corrected paths and hash/comparison receipts per camera,
reuse unchanged v7 raw evidence for counter/cache audit. Do not reselect cameras/pixels/triangles.
This is a correctness repair, not a repeated performance measurement or relaxed gate.
If any additional difference occurs, retain it and block promotion until explained.
