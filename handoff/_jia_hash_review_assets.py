#!/usr/bin/env python3
import hashlib
import pathlib

root = pathlib.Path("/mnt/c/Users/Someone/Desktop/SuGaR-main")
base = root / "unified_geometry_gs/output_compare/ggs_trust_rebuild_20261116"
paths = [
    "scan24/control_trunc4/vis/results.json",
    "scan24/control_trunc4/recon_post.ply",
    "scan24/control_trunc4/recon.ply",
    "scan24/control_trunc4/recon_aligned.ply",
    "scan24/model/point_cloud/iteration_30000/point_cloud.ply",
    "scan105/model/point_cloud/iteration_30000/point_cloud.ply",
    "scan122/model/point_cloud/iteration_30000/point_cloud.ply",
    "scan24/smoke/smoke_receipt.json",
    "scan24/smoke/smoke_arrays.npz",
    "scan24/smoke/smoke_redmask.png",
    "scan24/logs/impl_diff.patch",
]
for rel in paths:
    p = base / rel
    print(f"{rel}\t{p.stat().st_size}\t{hashlib.sha256(p.read_bytes()).hexdigest()}")
smoke = root / "unified_geometry_gs/scripts/smoke_jia_trust_fuse_scan24.py"
print(f"scripts/smoke_jia_trust_fuse_scan24.py\t{smoke.stat().st_size}\t{hashlib.sha256(smoke.read_bytes()).hexdigest()}")
