#!/usr/bin/env python3
"""G1 full: τ curve on scan24/105/122 (RESEARCH_PROTOCOL §5)."""
from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

SIGNIFICANT = 5e-4
PSNR_FLOOR = -0.2


def overall_for(tau: float, scan: str, out_root: Path) -> float | None:
    if abs(tau - 0.9) < 1e-9:
        p = out_root / "ggs_ex_trunc4" / scan / "vis" / "results.json"
    else:
        p = None
        for t in (f"ggs_tau_{tau:g}", f"ggs_tau_{tau}", f"ggs_tau_{tau:.1f}"):
            cand = out_root / t / scan / "vis" / "results.json"
            if cand.exists():
                p = cand
                break
        if p is None:
            return None
    if not p.exists():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["overall"])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--scans", nargs="+", default=["scan24", "scan105", "scan122"])
    ap.add_argument("--taus", nargs="+", type=float, default=[0.3, 0.5, 0.9, 1.2])
    ap.add_argument("--out_root", type=Path, default=Path("./output_compare"))
    ap.add_argument("--out", type=Path, default=Path("./output_compare/metrics/g1_verdict.json"))
    args = ap.parse_args()

    table: dict[str, dict[str, float | None]] = {}
    for scan in args.scans:
        table[scan] = {str(tau): overall_for(tau, scan, args.out_root) for tau in args.taus}

    c: dict[float, float | None] = {}
    for tau in args.taus:
        vals = [table[s][str(tau)] for s in args.scans if table[s][str(tau)] is not None]
        c[tau] = sum(vals) / len(vals) if len(vals) == len(args.scans) else None

    missing = [
        f"{s}@τ={t}"
        for s in args.scans
        for t in args.taus
        if table[s][str(t)] is None
    ]
    if missing or any(v is None for v in c.values()):
        out = {
            "protocol": "RESEARCH_PROTOCOL §5 G1",
            "verdict": "INCOMPLETE",
            "reason": f"missing: {missing}",
            "next": "finish P4 trains",
            "C_tau": {str(k): v for k, v in c.items()},
            "table": table,
        }
    else:
        ordered = sorted(args.taus)
        best_tau = min(args.taus, key=lambda t: c[t])  # type: ignore[arg-type]
        c_lo, c_hi = c[ordered[0]], c[ordered[-1]]
        c_best = c[best_tau]
        assert c_lo is not None and c_hi is not None and c_best is not None
        ends_higher = (c_lo >= c_best + SIGNIFICANT) and (c_hi >= c_best + SIGNIFICANT)
        interior = best_tau in (0.5, 0.9)
        rng = max(c[t] for t in args.taus) - min(c[t] for t in args.taus)  # type: ignore[type-var]

        if rng < SIGNIFICANT:
            verdict = "M1_no_write"
            reason = f"C(τ) range {rng:.6e} < {SIGNIFICANT}"
            next_step = "keep τ=0.9; M1 not in Method"
        elif ends_higher and interior:
            # Must also beat Pipeline* gate if τ* != 0.9
            if abs(best_tau - 0.9) < 1e-9:
                verdict = "M1_ok"
                reason = (
                    f"U-shape τ*=0.9; C_lo={c_lo:.6f} C*={c_best:.6f} C_hi={c_hi:.6f}"
                )
                next_step = "Pipeline* stays τ=0.9+trunc4; M1 claim OK; consider P5 full DTU"
            else:
                # compare mean overall vs 0.9
                delta = c_best - c[0.9]  # type: ignore[operator]
                if delta <= -SIGNIFICANT:
                    verdict = "M1_ok_replace"
                    reason = (
                        f"U-shape τ*={best_tau} beats 0.9 by {-delta:.6e}; "
                        f"C*={c_best:.6f}"
                    )
                    next_step = (
                        f"replace default τ→{best_tau} only after PSNR gate; then P5"
                    )
                else:
                    verdict = "M1_ok_keep_default"
                    reason = (
                        f"U-shape τ*={best_tau} but not clearly better than 0.9 "
                        f"(Δ={delta:+.6e}); curve OK, default stays 0.9"
                    )
                    next_step = "keep τ=0.9; publish curve; P5 optional"
        elif best_tau == ordered[0] or best_tau == ordered[-1]:
            delta = abs(c_best - c[0.9])  # type: ignore[arg-type, operator]
            if delta >= SIGNIFICANT:
                verdict = "M1_side"
                reason = f"single-sided; best τ*={best_tau}; |Δ vs 0.9|={delta:.6e}"
                next_step = "write single-sided M1; default stays 0.9 unless gate passes"
            else:
                verdict = "M1_no_write"
                reason = f"endpoint best but tiny Δ vs 0.9 ({delta:.6e})"
                next_step = "keep τ=0.9; demote M1"
        else:
            verdict = "M1_no_write"
            reason = f"no clear U; τ*={best_tau}; C={c}"
            next_step = "keep τ=0.9; demote M1"

        out = {
            "protocol": "RESEARCH_PROTOCOL §5 G1",
            "verdict": verdict,
            "reason": reason,
            "next": next_step,
            "tau_star": best_tau,
            "C_tau": {str(k): v for k, v in c.items()},
            "table": table,
            "significant": SIGNIFICANT,
            "psnr_note": "PSNR gate not re-checked here (extract-fixed trains); "
            "run compare_models if replacing τ default",
            "g2_note": "M2-emp; trunc4 empirical",
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"VERDICT: {out['verdict']}")
    print(f"reason: {out['reason']}")
    print(f"C(τ): {out.get('C_tau')}")
    print(f"NEXT: {out['next']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
