# Follow-up to the failed shared finite-difference gate

Written after shared_v1. Original shared FD h1e-4/h1e-5 failures remain
unchanged (~5.35% relative at1e-5), and branches losing roots retain
undefined means. No adoption/training claim follows from a diagnostic.

Use the exact saved common full direction/ID map, recompute all891 rays
at +/-h for h1e-7,1e-9,1e-11, without changing the loss or direction.
Record complete arrays, root statuses, original predicted derivative and
FD errors under the original tolerance. Audit decomposition of the old
h1e-5 discrepancy by individual ray. The largest contributor is
scan37_0034 pixel214675 (old anomaly1), chosen post hoc for diagnosis.
For that one fixed shared direction, independent240-digit composed MP
forward and root at the same three step sizes compare against the
existing analytic MP raw gradient contraction. No cap changes may be
silently treated as smooth. This selected-point result diagnoses the
observed error; it is not an independent confirmation or MP verification
of the full891-ray sum. Preserve failed Float64 micro steps too.
