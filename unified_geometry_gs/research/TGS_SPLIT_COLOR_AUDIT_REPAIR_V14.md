# Scalar-reader compatibility repair; no renderer/input/gate changes

2026-10-21. Renderer session22346 completed76 forwards and32 cases (exit0).
First CPU audit exited1 in `float(inp['sigma'])`: saved sigma/opacity are size-one
arrays, while NumPy2.4 permits Python float conversion only for ndim0. This is
a reader API error, not a failed numerical comparison or an observation timeout.

Preserve the original auditor, frozen plan, rendered outputs and traceback.
`audit_tgs_split_color_v14_v2.py` changes the two size-one scalar reads to `.item()`
and writes `independent_audit_v2.json`. Its dependency/source and input hashes are
recorded before auditing. All original gates, samples and metrics stay identical.
The original auditor remains in the renderer's plan and is still hash-verified.
No CUDA rerender, parameter change or threshold relaxation.
