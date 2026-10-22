#!/usr/bin/env python3
"""G1-pilot: τ curve on scan24+scan122 (RESEARCH_PROTOCOL §5)."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

SIGNIFICANT = 5e-4


def overall_for(tau: float, scan: str, out_root: Path) -> float | None:
    if abs(tau - 0.9) < 1e-9:
        p = out_root / "ggs_ex_trunc4" / scan / "vis" / "results.json"
    else:
        tag = f"ggs_tau_{tau:g}"
        # also try formatted with one decimal
        for t in (tag, f"ggs_tau_{tau}", f"ggs_tau_{tau:.1f}"):
            p = out_root / t / scan / "vis" / "results.json"
            if p.exists():
                break
        else:
            return None
    if not p.exists():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["overall"])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--scans", nargs="+", default=["scan24", "scan122"])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.3, 0.5, 0.9, 1.2])
    ap.add_argument("--out_root", type=Path, default=Path("./output_compare"))
    ap.add_argument("--out", type=Path, default=Path("./output_compare/metrics/g1_pilot_verdict.json"))
    args = ap.parse_args()

    table = {}
    for scan in args.scans:
        table[scan] = {}
        for tau in args.taus:
            table[scan][str(tau)] = overall_for(tau, scan, args.out_root)

    # Mean C(τ) over available scans
    c = {}
    for tau in args.taus:
        vals = [table[s][str(tau)] for s in args.scans if table[s][str(tau)] is not None]
        c[tau] = sum(vals) / len(vals) if vals else None

    missing = [f"{s}@τ={t}" for s in args.scans for t in args.taus if table[s][str(t)] is None]
    if missing:
        verdict = "INCOMPLETE"
        reason = f"missing overalls: {missing}"
        next_step = "finish P3 trains"
    else:
        # U-shape: min at 0.5 or 0.9; both ends higher by SIGNIFICANT
        ordered = sorted(args.taus)
        best_tau = min(args.taus, key=lambda t: c[t])
        c_lo, c_hi = c[ordered[0]], c[ordered[-1]]
        c_best = c[best_tau]
        ends_higher = (c_lo >= c_best + SIGNIFICANT) and (c_hi >= c_best + SIGNIFICANT)
        interior = best_tau in (0.5, 0.9)
        rng = max(c[t] for t in args.taus) - min(c[t] for t in args.taus)

        if rng < SIGNIFICANT:
            verdict = "no_signal"
            reason = f"C(τ) range {rng:.6e} < {SIGNIFICANT}"
            next_step = "stop training side; paper geometry story = M2-emp only"
        elif ends_higher and interior:
            verdict = "shape_ok"
            reason = (
                f"U-shape: τ*={best_tau}, "
                f"C_lo={c_lo:.6f}, C*={c_best:.6f}, C_hi={c_hi:.6f}"
            )
            next_step = "P4: add scan105 for same τ grid"
        elif best_tau == ordered[0] or best_tau == ordered[-1]:
            # monotonic-ish
            verdict = "shape_fail_mono"
            reason = f"best at endpoint τ*={best_tau}; C={c}"
            next_step = "stop τ training; M1 single-sided only if |C(best)-C(0.9)|>=5e-4"
        else:
            verdict = "shape_fail"
            reason = f"best τ*={best_tau} not in {{0.5,0.9}} or ends not higher; C={c}"
            next_step = "stop τ training; demote M1"

    out = {
        "protocol": "RESEARCH_PROTOCOL §5 G1-pilot",
        "verdict": verdict,
        "reason": reason,
        "next": next_step,
        "C_tau": {str(k): v for k, v in c.items()},
        "table": table,
        "significant": SIGNIFICANT,
        "g2_note": "G2 was M2-emp; trunc4 remains empirical extract default",
    }
    # fix reason if shape_ok with dict in f-string issue
    if verdict == "shape_ok":
        out["reason"] = (
            f"U-shape: τ*={best_tau}, "
            f"C_lo={c_lo:.6f}, C*={c_best:.6f}, C_hi={c_hi:.6f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"VERDICT: {verdict}")
    print(f"reason: {out['reason']}")
    print(f"C(τ): {c}")
    print(f"NEXT: {next_step}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
