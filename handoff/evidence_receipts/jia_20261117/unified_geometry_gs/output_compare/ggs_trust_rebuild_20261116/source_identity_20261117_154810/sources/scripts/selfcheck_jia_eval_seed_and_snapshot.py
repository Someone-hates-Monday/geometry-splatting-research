#!/usr/bin/env python3
"""Mechanical preflight: strict eval_seed validation + source snapshot vs compat receipt."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_dtu_mesh import assert_results_eval_seed  # noqa: E402


def expect_fail(label: str, fn) -> None:
    try:
        fn()
    except RuntimeError as e:
        print(f"OK_FAIL {label}: {e}")
        return
    raise SystemExit(f"expected failure for {label}")


def main() -> None:
    # --- strict seed field cases ---
    assert_results_eval_seed({"eval_seed": 0, "overall": 1.0}, 0)
    print("OK_PASS correct_seed_0")
    expect_fail("missing", lambda: assert_results_eval_seed({"overall": 1.0}, 0))
    expect_fail("null", lambda: assert_results_eval_seed({"eval_seed": None, "overall": 1.0}, 0))
    expect_fail("wrong", lambda: assert_results_eval_seed({"eval_seed": 1, "overall": 1.0}, 0))
    # When parent did not request a seed, do not require field
    assert_results_eval_seed({"overall": 1.0}, None)
    print("OK_PASS no_seed_requested")

    if len(sys.argv) < 3:
        raise SystemExit("usage: selfcheck_jia_eval_seed_and_snapshot.py <manifest.json> <compat_receipt.json>")
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    compat = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    loss_now = None
    for e in manifest["entries"]:
        if e["relpath"] == "utils/loss_utils.py":
            loss_now = e["sha256"]
        p = ROOT / e["relpath"]
        dig = hashlib.sha256(p.read_bytes()).hexdigest()
        if dig != e["sha256"]:
            raise SystemExit(f"manifest stale for {e['relpath']}: file={dig} manifest={e['sha256']}")
        snap = Path(sys.argv[1]).parent / e["snapshot_relpath"]
        if hashlib.sha256(snap.read_bytes()).hexdigest() != dig:
            raise SystemExit(f"snapshot copy mismatch {e['relpath']}")
    expected = compat["baseline"]["current_loss_utils_sha256"]
    if loss_now != expected:
        raise SystemExit(
            f"loss_utils sha mismatch vs compat receipt:\n  now={loss_now}\n  compat={expected}\n"
            "Re-run compat if loss graph changed."
        )
    # eval.py must be present and marked ignored
    eval_e = next(e for e in manifest["entries"] if e["relpath"] == "dtu_eval/eval.py")
    if eval_e.get("git_ignored") is not True:
        raise SystemExit(f"expected dtu_eval/eval.py git_ignored=true, got {eval_e.get('git_ignored')}")
    helpers = ["utils/jia_io_guard.py", "scripts/compat_jia_patchmatch_grad.py", "scripts/smoke_jia_trust_fuse_scan24.py"]
    for h in helpers:
        if not any(e["relpath"] == h for e in manifest["entries"]):
            raise SystemExit(f"manifest missing helper {h}")
    print(json.dumps({"ok": True, "loss_utils_sha256": loss_now, "eval_ignored": True}, indent=2))


if __name__ == "__main__":
    main()
