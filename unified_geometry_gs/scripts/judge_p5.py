#!/usr/bin/env python3
"""P5: full-DTU sign check for Pipeline* vs G-GS baseline (RESEARCH_PROTOCOL §5)."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path


def load_overall(p: Path) -> float | None:
    if not p.exists():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["overall"])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--scans", nargs="+", required=True)
    ap.add_argument("--pipeline_root", type=Path, default=Path("./output_compare/ggs_pipeline"))
    ap.add_argument("--pilot_root", type=Path, default=Path("./output_compare/ggs_ex_trunc4"))
    ap.add_argument("--baseline_root", type=Path, default=Path("./output_compare/ggs_baseline"))
    ap.add_argument("--out", type=Path, default=Path("./output_compare/metrics/p5_fulldtu_verdict.json"))
    args = ap.parse_args()

    pilot = {"scan24", "scan105", "scan122"}
    rows = []
    for scan in args.scans:
        if scan in pilot:
            pipe_p = args.pilot_root / scan / "vis" / "results.json"
            src = "pilot_trunc4"
        else:
            pipe_p = args.pipeline_root / scan / "vis" / "results.json"
            src = "ggs_pipeline"
        base_p = args.baseline_root / scan / "vis" / "results.json"
        po = load_overall(pipe_p)
        bo = load_overall(base_p)
        delta = (po - bo) if (po is not None and bo is not None) else None
        rows.append(
            {
                "scan": scan,
                "source": src,
                "pipeline_overall": po,
                "baseline_overall": bo,
                "delta_pipeline_minus_baseline": delta,
                "sign": (
                    "better"
                    if delta is not None and delta < 0
                    else ("worse" if delta is not None and delta > 0 else "n/a")
                ),
            }
        )

    paired = [r for r in rows if r["delta_pipeline_minus_baseline"] is not None]
    n_better = sum(1 for r in paired if r["delta_pipeline_minus_baseline"] < 0)
    n_worse = sum(1 for r in paired if r["delta_pipeline_minus_baseline"] > 0)
    n_paired = len(paired)
    mean_delta = (
        sum(r["delta_pipeline_minus_baseline"] for r in paired) / n_paired if n_paired else None
    )

    # Protocol: if >1/3 scans reverse sign vs pilot expectation (pilot mean was better),
    # demote claim. Pilot expectation: Pipeline* better (negative Δ).
    frac_worse = (n_worse / n_paired) if n_paired else 1.0
    if n_paired == 0:
        verdict = "INCOMPLETE"
        reason = "no paired baseline comparisons"
        next_step = "need baseline meshes or skip P5 table"
    elif frac_worse > 1.0 / 3.0:
        verdict = "CLAIM_SUBSET"
        reason = (
            f"{n_worse}/{n_paired} scans worse than baseline (>{1/3:.0%}); "
            f"mean Δ={mean_delta:+.6e}"
        )
        next_step = "claim holds on object-scale pilot subset; list failing scans in paper"
    else:
        verdict = "CLAIM_HOLDS"
        reason = (
            f"{n_better}/{n_paired} better, {n_worse} worse; mean Δ={mean_delta:+.6e}"
        )
        next_step = "Pipeline* + M1 U-curve + M2-emp trunc4 ready for write-up"

    out = {
        "protocol": "RESEARCH_PROTOCOL §5 P5",
        "verdict": verdict,
        "reason": reason,
        "next": next_step,
        "n_paired": n_paired,
        "n_better": n_better,
        "n_worse": n_worse,
        "frac_worse": frac_worse,
        "mean_delta_overall": mean_delta,
        "rows": rows,
        "recipe": "Train(no_dn,τ=0.9) → Extract(trunc_voxel_multiplier=4)",
        "g1": "M1_ok τ*=0.9",
        "g2": "M2-emp",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"VERDICT: {verdict}")
    print(f"reason: {reason}")
    print(f"NEXT: {next_step}")
    for r in rows:
        d = r["delta_pipeline_minus_baseline"]
        dstr = f"{d:+.6e}" if d is not None else "n/a"
        print(
            f"  {r['scan']}: pipe={r['pipeline_overall']} base={r['baseline_overall']} "
            f"Δ={dstr} ({r['sign']})"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
