"""Opt-in frozen-model mesh extraction with an isolated four-pass depth solver.

Validated on three frozen DTU models; does not change training defaults or the
installed CUDA package. Fresh output required; original metadata is hash-checked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.benchmark_median_cuda import load_config
from research.median_cuda_backend import activate, load_backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("four", "control"), default="four")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--voxel_size", type=float, default=.002)
    parser.add_argument("--trunc_voxel_multiplier", type=float, default=4.)
    parser.add_argument("--depth_max", type=float, default=8.)
    parser.add_argument("--num_cluster", type=int, default=1)
    args = parser.parse_args()
    args.model, args.out = args.model.resolve(), args.out.resolve()
    if args.out.exists() or min(args.iteration,args.voxel_size,args.trunc_voxel_multiplier,args.depth_max,args.num_cluster) <= 0:
        parser.error("Fresh output and positive extraction parameters required.")
    ply = args.model / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    if not ply.is_file() or not (args.model / "cfg_args").is_file():
        parser.error("Model PLY and cfg_args required.")
    originals = [ply] + [args.model / name for name in ("cfg_args", "cameras.json", "multi_view.json") if (args.model / name).is_file()]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "point_cloud").symlink_to(args.model / "point_cloud", target_is_directory=True)
    (args.out / "cfg_args").write_bytes((args.model / "cfg_args").read_bytes())
    state = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "building",
        "variant": args.variant, "original_model": str(args.model), "iteration": args.iteration,
        "parameters": {k:getattr(args,k) for k in ("voxel_size","trunc_voxel_multiplier","depth_max","num_cluster")},
        "original_file_sha256": hashes,
        "limitations": ["Frozen mesh extraction only; no training-gradient or novel-view improvement claim.",
            "Four-pass efficiency/quality evidence currently covers three DTU models and 24 timed views.",
            "Custom voxel/truncation values are not covered by that geometry comparison.",
            "Output point_cloud is a symlink to the original; this wrapper treats it as read-only."]}
    manifest = args.out / "extraction_manifest.json"
    try:
        manifest.write_text(json.dumps(state,indent=2)+"\n")
        backend = load_backend(args.variant)
        state.update(status="extracting", backend=str(backend.__file__),
            backend_sha256=hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest())
        manifest.write_text(json.dumps(state,indent=2)+"\n")
        dataset, pipe = load_config(args.model)
        dataset.model_path = str(args.out)
        from mesh_extract import extract_mesh
        started = time.perf_counter()
        with torch.no_grad(), activate(backend):
            extract_mesh(dataset,pipe,args.iteration,num_cluster=args.num_cluster,voxel_size=args.voxel_size,
                depth_max=args.depth_max,trunc_voxel_multiplier=args.trunc_voxel_multiplier,write_dir=str(args.out))
        state["extract_wall_seconds_including_load"] = time.perf_counter()-started
        for name,digest in hashes.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Original model file changed: {name}")
        state.update(status="complete", original_files_unchanged=True,
            mesh_sha256={name:hashlib.sha256((args.out/name).read_bytes()).hexdigest() for name in ("recon.ply","recon_post.ply")})
    except BaseException as error:
        state.update(status="failed",error=repr(error))
        raise
    finally:
        state["source_sha256"] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),ROOT/"research/median_cuda_backend.py",ROOT/"research/benchmark_median_cuda.py",ROOT/"mesh_extract.py")}
        manifest.write_text(json.dumps(state,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
