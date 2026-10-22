"""Run the real training entry point and record whether NCC was exercised.

Only the first NCC call is synchronized for diagnostics. Both arms use this
wrapper; runtime includes initialization, diagnostics and checkpoint writes.
"""
from __future__ import annotations

import atexit
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import warp_patch_ncc
from research import ncc_exact_backend


def main():
    args = sys.argv[1:]
    model = Path(args[args.index("-m") + 1])
    exact = "--ncc_exact_gradient" in args
    target = ncc_exact_backend if exact else warp_patch_ncc
    original = target.warp_patch_ncc
    started = time.perf_counter()
    report = {"backend": "exact" if exact else "legacy", "calls": 0,
              "first_call": None, "completed": False}

    def traced(*a, **kw):
        result = original(*a, **kw)
        report["calls"] += 1
        if report["first_call"] is None:
            score, valid = result
            first = {"patches": score.numel(), "valid": int(valid.sum().item()),
                     "accepted_tau_0_9": int((valid & ((1 - score).clamp(0, 2) < .9)).sum().item()),
                     "valid_scores_finite": bool(torch.isfinite(score[valid]).all().item())}
            report["first_call"] = first
            if score.requires_grad:
                def hook(gradient):
                    first["backward_finite"] = bool(torch.isfinite(gradient).all().item())
                    first["backward_nonzero"] = int((gradient != 0).sum().item())
                score.register_hook(hook)
        return result

    target.warp_patch_ncc = traced
    torch.cuda.reset_peak_memory_stats()

    def write_report():
        report["wall_seconds"] = time.perf_counter() - started
        report["peak_torch_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
        model.mkdir(parents=True, exist_ok=True)
        (model / "training_integration.json").write_text(json.dumps(report, indent=2) + "\n")
    atexit.register(write_report)
    sys.argv = [str(ROOT / "train.py"), *args]
    runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
    report["completed"] = True
    if report["calls"] == 0:
        raise RuntimeError("Training did not exercise the selected NCC backend.")


if __name__ == "__main__":
    main()
