# Confirmation v1 serialization failure retained

V1 completed MP references, GPU new-case arrays and both absent updates,
but failed while serializing a NumPy int64 pass count. The original
manifest/traceback/new_arrays.npz and exact source ZIP are retained in
history_gap_gpu_confirm_v1. No numerical failure was concluded from it.
V2 uses the identical models, seed, CUDA binary, expressions and gates,
with NumPy scalar serialization fixed. This does not alter the CUDA
implementation or any original experiment result.
