#!/usr/bin/env python3
import hashlib
import json
import pathlib

base = pathlib.Path(
    "/mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116/scan24"
)
stamp = "20261117_145006"
paths = [
    f"compat_grad_{stamp}/compat_receipt.json",
    f"smoke_{stamp}/smoke_receipt.json",
    f"smoke_{stamp}/smoke_arrays.npz",
    f"smoke_{stamp}/smoke_redmask.png",
    f"control_trunc4_eval_seed0_run1_{stamp}/vis/results.json",
    f"control_trunc4_eval_seed0_run2_{stamp}/vis/results.json",
    f"control_trunc4_eval_seed0_run1_{stamp}/eval_receipt.json",
    "control_trunc4/vis/results.json",
    f"logs/p1_remediation_summary_{stamp}.json",
    "logs/impl_diff_p1_remediation.patch",
]
for rel in paths:
    p = base / rel
    print(f"{rel}\t{p.stat().st_size}\t{hashlib.sha256(p.read_bytes()).hexdigest()}")

compat = json.loads((base / f"compat_grad_{stamp}/compat_receipt.json").read_text())
print("compat_ok", compat["ok"])
for c in compat["cases"]:
    print(" case", c["case"], c["ok"])
smoke = json.loads((base / f"smoke_{stamp}/smoke_receipt.json").read_text())
st = smoke["run1"]["stats"]
print(
    "smoke_two_run",
    smoke["two_run_consistent"],
    "keep",
    st["frac_keep"],
    "subdiv_ok",
    st.get("other_fail_subdiv_sum_eq"),
    "geom",
    st.get("n_geom_fail"),
    "warp",
    st.get("n_warp_fail"),
    "nfe",
    st.get("n_nonfinite_error"),
)

# untracked helpers
root = pathlib.Path("/mnt/c/Users/Someone/Desktop/SuGaR-main/unified_geometry_gs")
for rel in [
    "utils/jia_io_guard.py",
    "dtu_eval/eval.py",
    "scripts/compat_jia_patchmatch_grad.py",
    "scripts/smoke_jia_trust_fuse_scan24.py",
]:
    p = root / rel
    print(f"SRC {rel}\t{p.stat().st_size}\t{hashlib.sha256(p.read_bytes()).hexdigest()}")
