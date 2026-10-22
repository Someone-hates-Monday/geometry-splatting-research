"""Re-evaluate a frozen aligned mesh with a seeded DTU evaluator copy.

Preserves the original evaluator, alignment, metrics and previous outputs.
Only shuffle seed and CPU concurrency are changed in a recorded source copy.
This is evaluation replication, not a new training repetition.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scan", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dtu_eval/Offical_DTU_Dataset")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.mesh.is_file() or min(args.scan, args.workers) < 1 or args.seed < 0:
        parser.error("Existing aligned mesh, positive scan/workers and nonnegative seed required.")
    args.mesh, args.dataset, args.out = (p.resolve() for p in (args.mesh, args.dataset, args.out))
    source_path = ROOT / "dtu_eval/eval.py"
    source = source_path.read_text()
    changes = {
        "shuffle_rng = np.random.default_rng()": f"shuffle_rng = np.random.default_rng({args.seed})",
        "with mp.Pool() as mp_pool:": f"with mp.Pool(processes={args.workers}) as mp_pool:",
        "n_jobs=-1": "n_jobs=1",
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ValueError(f"Expected exactly one source target: {old}")
        source = source.replace(old, new)
    gt_paths = [args.dataset / "ObsMask" / f"ObsMask{args.scan}_10.mat",
                args.dataset / "ObsMask" / f"Plane{args.scan}.mat",
                args.dataset / "Points/stl" / f"stl{args.scan:03d}_total.ply"]
    if not all(p.is_file() for p in gt_paths):
        raise FileNotFoundError("Missing official DTU ground-truth files.")
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / "eval_seeded_snapshot.py"
    snapshot.write_text(source, encoding="utf-8")
    command = [sys.executable, str(snapshot), "--data", str(args.mesh), "--scan", str(args.scan),
               "--mode", "mesh", "--dataset_dir", str(args.dataset), "--vis_out_dir", str(args.out)]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "scan": args.scan, "shuffle_seed": args.seed, "workers": args.workers, "neighbor_jobs": 1,
        "argv": command, "changes": changes,
        "source_sha256": {str(p): digest(p) for p in [Path(__file__), source_path, snapshot, args.mesh] + gt_paths},
        "environment": {"python": sys.version, **{p: version(p) for p in ("numpy", "scipy", "scikit-learn", "open3d")}},
        "limitations": ["Same training checkpoint and aligned mesh; evaluation repetition only.",
            "Matched shuffle seeds do not imply matched candidate points across different meshes.",
            "Original official-style unseeded evaluation outputs are preserved separately.",
            "CPU concurrency changed for resource control; this run is not a speed benchmark."]}
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        with (args.out / "evaluate.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        manifest["returncode"] = process.returncode
        if process.returncode:
            raise RuntimeError("Seeded DTU evaluator failed; see evaluate.log.")
        result = json.loads((args.out / "results.json").read_text())
        if not all(math.isfinite(result[k]) for k in ("mean_d2s", "mean_s2d", "overall")):
            raise ValueError("Nonfinite mesh metrics.")
        if abs(result["overall"] - .5 * (result["mean_d2s"] + result["mean_s2d"])) > 1e-12:
            raise ValueError("Overall does not match the two directed distances.")
        manifest.update(status="complete", metrics=result)
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["wall_seconds"] = time.perf_counter() - started
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
