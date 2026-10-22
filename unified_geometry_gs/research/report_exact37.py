"""Assemble a bounded conclusion from the completed NCC pilot, without retraining."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--seeded", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Fresh report path required.")
    sources = []

    def read(path):
        sources.append(path)
        return json.loads(path.read_text())

    manifest = read(args.pair / "manifest.json")
    post = read(args.post / "manifest.json")
    seeded_manifest = read(args.seeded / "manifest.json")
    if any(x["status"] != "complete" for x in (manifest, post, seeded_manifest)):
        raise ValueError("All three experiment stages must be complete.")
    metrics = read(args.pair / "paired_mesh_metrics.json")
    seeded = read(args.seeded / "paired_seeded_metrics.json")
    images, integrations = {}, {}
    for arm in ("legacy", "exact"):
        images[arm] = read(args.post / f"images_{arm}.json")
        integrations[arm] = read(args.pair / arm / "training_integration.json")
        if images[arm]["partial_smoke"] or images[arm]["is_novel_view_evaluation"]:
            raise ValueError("This report expects complete training-view evaluation.")
        if len(images[arm]["rows"]) != images[arm]["camera_count"]:
            raise ValueError("Camera count mismatch.")
        for key, value in images[arm]["mean"].items():
            if abs(statistics.mean(row[key] for row in images[arm]["rows"]) - value) > 1e-10:
                raise ValueError("Image aggregate mismatch.")
    if [r["camera"] for r in images["legacy"]["rows"]] != [r["camera"] for r in images["exact"]["rows"]]:
        raise ValueError("Image camera lists differ.")
    deltas = [x["delta_exact_minus_legacy"]["overall"] for x in seeded["paired_evaluations"]]
    image_delta = {k: images["exact"]["mean"][k] - images["legacy"]["mean"][k] for k in images["legacy"]["mean"]}
    first_calls = {arm: integrations[arm]["first_call"] for arm in integrations}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scene": manifest["scan"],
        "decision": "Do not promote exact NCC gradient as a reconstruction improvement from this pilot.",
        "unseeded_mesh_evaluation": metrics,
        "fixed_evaluator_seeds": [x["evaluation_seed"] for x in seeded["paired_evaluations"]],
        "seeded_mean_delta": seeded["mean_delta"],
        "seeded_overall_delta_range": [min(deltas), max(deltas)],
        "all_seeded_overall_deltas_positive": all(x > 0 for x in deltas),
        "seed0_repeat_numeric_equal": seeded["seed0_repeat_bitwise_numeric_equal"],
        "training_view_image_means": {arm: images[arm]["mean"] for arm in images},
        "training_view_image_delta": image_delta,
        "cameras": [r["camera"] for r in images["legacy"]["rows"]],
        "gaussian_count": {arm: images[arm]["gaussian_count"] for arm in images},
        "gaussian_ply_bytes": {arm: images[arm]["gaussian_ply_bytes"] for arm in images},
        "first_ncc_calls": first_calls,
        "first_call_patch_counts_equal": first_calls["legacy"]["patches"] == first_calls["exact"]["patches"],
        "causal_limit": "The first NCC input patch counts differ before either NCC backward has run. Equal initial seeds did not give identical warmup states; this pair does not isolate the derivative's causal effect.",
        "limitations": [
            "One scene and one training seed per arm; evaluator repeats are not training replications.",
            "All 49 cameras are training views; no novel-view generalization result.",
            "Image improvement and d2s improvement coexist with worse s2d/overall; no composite score invented afterward.",
            "Sequential single-run training wall clocks and Torch memory peaks do not establish acceleration.",
            "A shared-warmup branch experiment needs optimizer/appearance/RNG/camera-order restoration checks before use.",
        ],
        "source_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources + [Path(__file__)]},
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("decision", "seeded_mean_delta", "training_view_image_delta", "first_call_patch_counts_equal")}, indent=2))


if __name__ == "__main__":
    main()
