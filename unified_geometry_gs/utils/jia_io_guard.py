"""Refuse overwriting existing mesh/eval artifacts on explicit new-path writes."""
from __future__ import annotations

import os
from typing import Iterable, Sequence


DEFAULT_MESH_ARTIFACTS: tuple[str, ...] = (
    "recon.ply",
    "recon_post.ply",
    "recon_aligned.ply",
    "per_view_metrics.json",
    os.path.join("vis", "results.json"),
)

DEFAULT_EVAL_ARTIFACTS: tuple[str, ...] = (
    "recon_aligned.ply",
    os.path.join("vis", "results.json"),
    os.path.join("vis", "vis_meta.json"),
)


def _abspath(path: str) -> str:
    return os.path.abspath(os.path.normpath(path))


def _is_same_or_nested(path: str, root: str) -> bool:
    path_a = _abspath(path)
    root_a = _abspath(root)
    if path_a == root_a:
        return True
    prefix = root_a if root_a.endswith(os.sep) else root_a + os.sep
    return path_a.startswith(prefix)


def assert_fresh_output_dir(
    write_dir: str,
    *,
    artifacts: Sequence[str],
    forbidden_roots: Iterable[str] | None = None,
    label: str = "output",
) -> str:
    """Require an explicit write dir with no protected artifacts and no path conflicts.

    Legacy callers that write into model_path without an explicit write_dir should
    not call this (keeps old entry behavior).
    """
    if not write_dir:
        raise ValueError(f"{label}: empty write_dir")
    target = _abspath(write_dir)
    for root in forbidden_roots or ():
        if not root:
            continue
        if _is_same_or_nested(target, root) or _is_same_or_nested(root, target):
            raise RuntimeError(
                f"{label}: refuse write_dir={target!r} conflicting with { _abspath(root)!r }"
            )
    existing = []
    for rel in artifacts:
        p = os.path.join(target, rel)
        if os.path.exists(p):
            existing.append(p)
    if existing:
        raise RuntimeError(
            f"{label}: refuse overwrite; already exists:\n  " + "\n  ".join(existing)
        )
    os.makedirs(target, exist_ok=True)
    return target
