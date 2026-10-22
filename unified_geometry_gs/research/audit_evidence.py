"""Recompute descriptive evidence without altering historical results/verdicts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
PILOT = ("scan24", "scan105", "scan122")
MESH_KEYS = ("overall", "mean_d2s", "mean_s2d")


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def paired_stage(metrics, pattern):
    rows = []
    for scan in PILOT:
        path = metrics / pattern.format(scan=scan)
        record = read(path)
        if record:
            rows.append({"scan": scan, "source": str(path.relative_to(ROOT)),
                         "delta": record["delta_optimized_minus_baseline"]})
    keys = (*MESH_KEYS, "psnr")
    average = {k: mean(r["delta"][k] for r in rows) for k in keys if rows and all(k in r["delta"] for r in rows)}
    gate = {
        "three_scans_present": len(rows) == len(PILOT),
        "mean_overall_pass": average.get("overall", float("inf")) <= -5e-4,
        "mean_s2d_pass": average.get("mean_s2d", float("inf")) <= 0,
        "mean_training_psnr_pass": average["psnr"] >= -.2 if "psnr" in average else None,
    }
    return {"rows": rows, "mean_delta": average, "gate_components": gate,
            "scope": "Existing comparison JSONs, with their original evaluation runs; no new evaluation."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "research/results/evidence_audit.json")
    args = parser.parse_args()
    outputs = ROOT / "output_compare"
    metrics = outputs / "metrics"
    scans = sorted(p.name for p in (ROOT.parent / "dtu/DTU").glob("scan*") if p.is_dir())
    rows = []
    sources = []
    for scan in scans:
        tag = "ggs_ex_trunc4" if scan in PILOT else "ggs_pipeline"
        pp = outputs / tag / scan / "vis/results.json"
        bp = outputs / "ggs_baseline" / scan / "vis/results.json"
        pipe, base = read(pp), read(bp)
        sources.extend(p for p in (pp, bp) if p.exists())
        rows.append({"scan": scan, "pipeline": pipe, "baseline": base,
                     "delta": {k: pipe[k] - base[k] for k in MESH_KEYS} if pipe and base else None,
                     "pilot_used_for_selection": scan in PILOT})
    paired = [r for r in rows if r["delta"] is not None]
    heldout = [r for r in paired if not r["pilot_used_for_selection"]]
    tau = read(metrics / "g1_verdict.json")
    audit = {
        "dataset_scan_count": len(scans), "paired_scan_count": len(paired),
        "missing_baseline": [r["scan"] for r in rows if r["baseline"] is None],
        "missing_pipeline": [r["scan"] for r in rows if r["pipeline"] is None],
        "mean_delta_paired": {k: mean(r["delta"][k] for r in paired) for k in MESH_KEYS},
        "n_better": sum(r["delta"]["overall"] < 0 for r in paired),
        "heldout_paired_scan_count": len(heldout),
        "mean_delta_heldout": {k: mean(r["delta"][k] for r in heldout) for k in MESH_KEYS},
        "rows": rows,
        "training_no_dn_vs_baseline": paired_stage(metrics, "{scan}_baseline_vs_ggs_ablate_no_dn.json"),
        "extraction_trunc4_vs_no_dn": paired_stage(metrics, "{scan}_no_dn_vs_ex_trunc4.json"),
        "tau_curve": tau,
        "limitations": [
            "15-scene full-DTU claim requires all paired baselines; incomplete coverage must be stated.",
            "Pipeline contrast changes training and extraction together; cannot identify either contribution alone.",
            "Existing image metrics measure training views, not novel-view generalization.",
            "Saved cfg_args contain model parameters but not the complete optimization recipe or training seed.",
            "The 5e-4 protocol margin is an adoption threshold, not a statistical significance test.",
            "Tau=1.2 is the no-correlation-filter endpoint for finite nonnegative squared-NCC scores.",
            "A U-shaped four-point mean does not prove monotone bias/variance or an underlying population law.",
            "Existing stochastic mesh evaluation draws differ between comparison JSONs and current results.json.",
        ],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("dataset_scan_count", "paired_scan_count", "missing_baseline", "mean_delta_paired", "heldout_paired_scan_count", "mean_delta_heldout")}, indent=2))
    print("stage training", audit["training_no_dn_vs_baseline"]["mean_delta"])
    print("stage extraction", audit["extraction_trunc4_vs_no_dn"]["mean_delta"])


if __name__ == "__main__":
    main()
