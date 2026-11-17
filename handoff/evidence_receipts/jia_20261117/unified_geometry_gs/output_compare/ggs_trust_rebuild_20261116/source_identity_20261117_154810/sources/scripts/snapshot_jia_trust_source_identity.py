#!/usr/bin/env python3
"""Freeze source identity for trust-rebuild (includes gitignored/untracked files).

Copies critical sources into an evidence snapshot dir and writes manifest with SHA256.
Does not modify .gitignore or training code.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

# Files that must be identity-frozen for this experiment (tracked + ignored + untracked).
SOURCE_RELPATHS = [
    "utils/loss_utils.py",
    "utils/jia_io_guard.py",
    "mesh_extract.py",
    "evaluate_dtu_mesh.py",
    "dtu_eval/eval.py",
    "scripts/compat_jia_patchmatch_grad.py",
    "scripts/smoke_jia_trust_fuse_scan24.py",
    "scripts/snapshot_jia_trust_source_identity.py",
    "scripts/selfcheck_jia_eval_seed_and_snapshot.py",
    "scripts/run_jia_trust_full_extract.py",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_ignored(rel: str) -> bool | None:
    try:
        r = subprocess.run(
            ["git", "check-ignore", "-v", f"unified_geometry_gs/{rel}"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            return True
        return False
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out is None:
        raise SystemExit("usage: snapshot_jia_trust_source_identity.py <out_dir>")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refuse non-empty snapshot dir: {out}")
    out.mkdir(parents=True, exist_ok=True)
    src_dir = out / "sources"
    src_dir.mkdir()

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    entries = []
    for rel in SOURCE_RELPATHS:
        src = ROOT / rel
        if not src.is_file():
            raise SystemExit(f"missing required source: {src}")
        dest = src_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        digest = sha256_file(src)
        assert sha256_file(dest) == digest
        entries.append(
            {
                "relpath": rel,
                "sha256": digest,
                "bytes": src.stat().st_size,
                "git_ignored": git_ignored(rel),
                "snapshot_relpath": str(Path("sources") / rel),
            }
        )

    # Optional: record compat baseline expectation
    expected_loss = None
    for e in entries:
        if e["relpath"] == "utils/loss_utils.py":
            expected_loss = e["sha256"]

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": head,
        "uggs_root": str(ROOT),
        "entries": entries,
        "loss_utils_sha256": expected_loss,
        "note": "Includes gitignored dtu_eval/eval.py and untracked helpers; not a substitute for git commit.",
    }
    (out / "source_identity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "manifest": str(out / "source_identity_manifest.json"), "loss_utils_sha256": expected_loss}, indent=2))


if __name__ == "__main__":
    main()
