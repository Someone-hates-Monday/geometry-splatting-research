"""After GPU postprocessing, replicate fixed-mesh DTU evaluation on CPU.

Seeds 0,1,2 are fixed before reading the paired quality outcome. A duplicate
legacy seed-0 run checks reproducibility of the isolated seeded evaluator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
    parser.add_argument("--after", type=Path, required=True, help="GPU postprocessing manifest to wait for.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.pair, args.after, args.out = (p.resolve() for p in (args.pair, args.after, args.out))
    initial_pair = json.loads((args.pair / "manifest.json").read_text())
    if initial_pair.get("smoke_only") or not args.after.is_file():
        parser.error("Existing formal pair and GPU postprocessing manifest required.")
    scan = int(initial_pair["scan"].removeprefix("scan"))
    args.out.mkdir(parents=True, exist_ok=False)
    scripts = [Path(__file__), ROOT / "research/evaluate_mesh_seeded.py", ROOT / "dtu_eval/eval.py"]
    state = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "waiting_for_gpu_postprocessing",
        "pair": str(args.pair), "after": str(args.after), "evaluation_seeds": [0, 1, 2], "steps": [],
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
        "scope": "Evaluation randomness of two fixed aligned meshes; not independent training seeds."}
    state_path = args.out / "manifest.json"
    dump(state_path, state)
    start_wait = time.monotonic()
    try:
        while True:
            prerequisite = json.loads(args.after.read_text())
            if prerequisite["status"] == "failed":
                raise RuntimeError("GPU postprocessing failed; seeded evaluation not started.")
            if prerequisite["status"] == "complete":
                break
            if time.monotonic() - start_wait > 6 * 3600:
                raise TimeoutError("GPU postprocessing did not finish within six hours.")
            time.sleep(15)
        for path, digest in state["source_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Evaluation source changed while waiting: {path}")
        state["status"] = "running"
        dump(state_path, state)
        jobs = [("legacy", 0, "legacy_seed0"), ("legacy", 0, "legacy_seed0_repeat")]
        jobs += [(arm, seed, f"{arm}_seed{seed}") for arm in ("legacy", "exact") for seed in (0, 1, 2)
                 if (arm, seed) != ("legacy", 0)]
        for arm, seed, name in jobs:
            command = [sys.executable, str(ROOT / "research/evaluate_mesh_seeded.py"),
                "--mesh", str(args.pair / arm / "recon_aligned.ply"), "--scan", str(scan),
                "--seed", str(seed), "--workers", "2", "--out", str(args.out / name)]
            step = {"stage": name, "argv": command, "status": "running"}
            state["steps"].append(step)
            dump(state_path, state)
            print(f"Starting {name}", flush=True)
            started = time.perf_counter()
            with (args.out / f"{name}.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            step.update(returncode=process.returncode, wall_seconds=time.perf_counter() - started,
                        status="complete" if process.returncode == 0 else "failed")
            dump(state_path, state)
            if process.returncode:
                raise RuntimeError(f"{name} failed; see its log.")
        load = lambda name: json.loads((args.out / name / "results.json").read_text())
        baseline, repeated = load("legacy_seed0"), load("legacy_seed0_repeat")
        repeated_delta = {k: repeated[k] - baseline[k] for k in ("overall", "mean_d2s", "mean_s2d")}
        rows = []
        for seed in (0, 1, 2):
            legacy, exact = load(f"legacy_seed{seed}"), load(f"exact_seed{seed}")
            rows.append({"evaluation_seed": seed, "legacy": legacy, "exact": exact,
                "delta_exact_minus_legacy": {k: exact[k] - legacy[k] for k in ("overall", "mean_d2s", "mean_s2d")}})
        result = {"seed0_repeat_delta": repeated_delta, "seed0_repeat_bitwise_numeric_equal": baseline == repeated,
            "paired_evaluations": rows, "mean_delta": {k: sum(r["delta_exact_minus_legacy"][k] for r in rows) / 3
                for k in ("overall", "mean_d2s", "mean_s2d")},
            "limitations": ["Three evaluator shuffles of the same mesh pair, not three training seeds.",
                "No confidence interval, statistical significance or novel-view generalization claim."]}
        dump(args.out / "paired_seeded_metrics.json", result)
        state["status"] = "complete"
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        dump(state_path, state)


if __name__ == "__main__":
    main()
