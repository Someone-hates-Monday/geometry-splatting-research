#!/usr/bin/env python3
"""
Aggregate compare_models JSONs from M1 ablation runs.

Expects filenames: <scan>_baseline_vs_<tag>.json
(e.g. scan105_baseline_vs_m1_ablate_03_full.json)

Prints per-tag mean delta (optimized - baseline) across scans.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize M1 ablation metrics JSONs.")
    p.add_argument(
        "--metrics_dir",
        type=str,
        default="./output_compare/metrics",
        help="Directory containing *_baseline_vs_*.json",
    )
    p.add_argument(
        "--tag_prefix",
        type=str,
        default="m1_ablate_",
        help="Only include files whose tag starts with this (after baseline_vs_).",
    )
    args = p.parse_args()
    root = Path(args.metrics_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    pat = re.compile(r"^(.+)_baseline_vs_(.+)\.json$")
    by_tag: dict[str, list[tuple[str, dict]]] = defaultdict(list)

    for path in sorted(root.glob("*.json")):
        m = pat.match(path.name)
        if not m:
            continue
        scan, tag = m.group(1), m.group(2)
        if not tag.startswith(args.tag_prefix):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"Skip {path.name}: {e}")
            continue
        delta = data.get("delta_optimized_minus_baseline")
        if not isinstance(delta, dict):
            continue
        by_tag[tag].append((scan, delta))

    if not by_tag:
        print(f"No matching JSON under {root} (prefix={args.tag_prefix!r})")
        return

    keys = ("overall", "mean_d2s", "mean_s2d", "l1", "psnr", "ssim")

    def mean(vals: list[float]) -> float | None:
        v = [x for x in vals if isinstance(x, (int, float))]
        return sum(v) / len(v) if v else None

    print(f"=== M1 ablation summary (n scans per tag) ===\nRoot: {root}\n")
    for tag in sorted(by_tag.keys()):
        rows = by_tag[tag]
        print(f"[{tag}]  n={len(rows)}")
        for scan, delta in sorted(rows, key=lambda x: x[0]):
            mesh = {k: delta.get(k) for k in ("overall", "mean_d2s", "mean_s2d")}
            img = {k: delta.get(k) for k in ("l1", "psnr", "ssim")}
            print(f"  {scan}: mesh {mesh} | img {img}")
        agg = {k: mean([r[1].get(k) for r in rows]) for k in keys}
        print(
            f"  AVG: overall={agg['overall']!s} d2s={agg['mean_d2s']!s} s2d={agg['mean_s2d']!s} | "
            f"l1={agg['l1']!s} psnr={agg['psnr']!s} ssim={agg['ssim']!s}"
        )
        print()


if __name__ == "__main__":
    main()
