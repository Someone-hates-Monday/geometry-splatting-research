# V1 confirmation failures and V2 exact-identity change

Initial CUDA real891/891 passes, but new synthetic32 only28/32 pass the
unchanged gates. Four zero-kernel aligned-ray cases have mathematically
zero xyz/quaternion blocks: tiny projected-residual and cancelling screen
AD roundoff fails their relative block gates (whole-ray errors~1e-14).
Do not replace these failures with a relaxed absolute gate.

V2 uses the exact identity E=det(F)^2||F^-T(v x d)||²/||Fd||² for
F=S^-1 R^-1, with F^-T=S R^T. Aligned world vectors have an exact zero
cross product in these fixtures, without projecting then subtracting.
For screen kernel0, gamma and all its derivatives are exactly zero,
so explicitly use that identity rather than subtracting equal log-dets.
The CUDA root/loss/seed kernel and all mathematical objectives are unchanged.

Recheck original891 and original synthetic32 as development regression.
Then check a separate fixed32 seed13019 with the same generation law,
nonzero kernels and gates; no training if any fails. Retain v1/v2/v3
confirmation records: v1 JSON scalar failure; v2 completed calculations
then final console scalar failure; v3 recovered immutable v2 artifacts
without rerunning GPU, showing the original28/32 verdict.
