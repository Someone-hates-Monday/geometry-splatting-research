"""Wait for one local paired run, then validate rays and evaluate both images.

No training code or pair manifest is modified. A failed/unfinished pair cannot
trigger evaluation, and each postprocessing run requires a new output folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout_hours", type=float, default=6)
    args = parser.parse_args()
    pair = args.pair.resolve()
    pair_manifest = pair / "manifest.json"
    if not pair_manifest.is_file() or args.timeout_hours <= 0:
        parser.error("An existing pair manifest and positive timeout are required.")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    scripts = (Path(__file__), ROOT / "research/validate_affine_cuda.py", ROOT / "research/evaluate_images.py")
    state = {"created_utc": datetime.now(timezone.utc).isoformat(), "pair": str(pair),
             "pid": os.getpid(), "status": "waiting_for_pair", "steps": [],
             "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
             "gpu_policy": "Do not start GPU work until the entire paired run has completed."}
    state_path = args.out / "manifest.json"
    dump(state_path, state)
    started = time.monotonic()
    previous_stage = None
    try:
        while True:
            # Pair writer is not atomic; a transient partial JSON is not a failed run.
            try:
                manifest = json.loads(pair_manifest.read_text())
            except json.JSONDecodeError:
                manifest = None
            if manifest:
                status = manifest.get("status")
                stage = (manifest.get("steps") or [{}])[-1].get("stage", status)
                if stage != previous_stage:
                    state["observed_pair_stage"] = stage
                    dump(state_path, state)
                    print(f"Waiting for pair: {stage} ({status})", flush=True)
                    previous_stage = stage
                if status == "failed":
                    raise RuntimeError(f"Pair failed: {manifest.get('error')}; no GPU postprocessing started.")
                if status == "complete":
                    if manifest.get("smoke_only") or not (pair / "paired_mesh_metrics.json").is_file():
                        raise ValueError("Postprocessing expects a complete formal pair with mesh metrics.")
                    for name, digest in manifest["source_sha256"].items():
                        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                            raise ValueError(f"Pair source changed during training: {name}")
                    state["pair_manifest_sha256_at_completion"] = hashlib.sha256(pair_manifest.read_bytes()).hexdigest()
                    break
            if time.monotonic() - started > args.timeout_hours * 3600:
                raise TimeoutError("Local pair did not complete within the waiting limit.")
            time.sleep(15)
        state["status"] = "running"
        dump(state_path, state)

        jobs = []
        for scan in ("scan37", "scan55", "scan63"):
            cpu = ROOT / f"research/results/median_cpu_{scan}.json"
            jobs.append((f"affine_{scan}", [sys.executable, str(ROOT / "research/validate_affine_cuda.py"),
                "--cpu", str(cpu), "--out", str(args.out / f"affine_cuda_{scan}.json")]))
        for arm in ("legacy", "exact"):
            jobs.append((f"images_{arm}", [sys.executable, str(ROOT / "research/evaluate_images.py"),
                "--model", str(pair / arm), "--out", str(args.out / f"images_{arm}.json")]))
        for name, command in jobs:
            step = {"stage": name, "argv": command, "status": "running"}
            state["steps"].append(step)
            dump(state_path, state)
            print(f"Starting {name}", flush=True)
            start = time.perf_counter()
            with (args.out / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            step.update(returncode=process.returncode, wall_seconds=time.perf_counter() - start,
                        status="complete" if process.returncode == 0 else "failed")
            dump(state_path, state)
            if process.returncode:
                raise RuntimeError(f"{name} failed; see {name}.log")
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        dump(state_path, state)


if __name__ == "__main__":
    main()
