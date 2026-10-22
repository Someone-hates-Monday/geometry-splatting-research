"""Descriptive paired-view summaries of completed frozen GPU measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Fresh output required.")
    data = json.loads(args.input.read_text())
    models = sorted({r["model"] for r in data["rows"]})
    scenes = []
    for model in models:
        rows = [r for r in data["rows"] if r["model"] == model]
        control = {r["camera"]: r for r in rows if r["variant"] == "control"}
        summaries = {}
        for variant in sorted({r["variant"] for r in rows}):
            selected = [r for r in rows if r["variant"] == variant]
            if {r["camera"] for r in selected} != set(control):
                raise ValueError("Variant camera lists differ.")
            ratios = [r["gpu_ms_median"]/control[r["camera"]]["gpu_ms_median"] for r in selected]
            summaries[variant] = {
                "views": len(selected), "mean_view_gpu_ms": statistics.mean(r["gpu_ms_median"] for r in selected),
                "median_view_latency_ratio_to_control": statistics.median(ratios),
                "view_latency_ratio_range_to_control": [min(ratios), max(ratios)],
                "full_image_validity_mismatches": sum(r["full_image_validity_mismatches"] for r in selected),
                "sampled_cpu_validity_mismatches": sum(r["sampled_cpu_validity_mismatches"] for r in selected),
                "sampled_cpu_errors_gt_1e_4": sum(r["sampled_depth_errors_gt_1e_4"] for r in selected),
                "max_full_image_depth_difference_vs_installed": max(r["depth_error_vs_installed"]["max"] for r in selected),
                "max_sampled_cpu_depth_error": max(r["sampled_depth_error_vs_cpu"]["max"] for r in selected),
                "rgb_max_abs_difference": max(r["rgb_max_abs_difference"] for r in selected),
                "alpha_max_abs_difference": max(r["alpha_max_abs_difference"] for r in selected),
                "normal_max_abs_difference": max(r["normal_max_abs_difference"] for r in selected),
                "depth_off_mean_view_gpu_ms": statistics.mean(r["depth_off_gpu_ms_median"] for r in selected),
            }
        scenes.append({"model": model, "variants": summaries})
    report = {"scenes": scenes, "source_sha256": {
        str(args.input.resolve()): hashlib.sha256(args.input.read_bytes()).hexdigest(),
        str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "interpretation": "Latency ratios <1 mean faster for these frozen renders. No training-speed or academic-novelty claim.",
        "uncertainty": "Descriptive view-level ranges; correlated cameras are not independent training replications.",
        "limitations": data["limitations"]}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(scenes, indent=2))


if __name__ == "__main__":
    main()
