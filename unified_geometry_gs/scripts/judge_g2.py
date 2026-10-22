#!/usr/bin/env python3
"""G2 judgment: t* vs σ_z → M2-const / M2-adapt / M2-emp (RESEARCH_PROTOCOL §5).

Reads absolute overall from each extract's vis/results.json (not cross-baseline Δ).
"""
from __future__ import annotations

import json
from pathlib import Path
from argparse import ArgumentParser

# m -> t = m * 0.002
VOXEL = 0.002
T_GRID_M = (2, 4, 6, 8, 12)
SIGNIFICANT = 5e-4


def load_overall(model_dir: Path) -> float | None:
    p = model_dir / "vis" / "results.json"
    if not p.exists():
        return None
    return float(json.loads(p.read_text(encoding="utf-8"))["overall"] )


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--sigma_z", type=Path, default=Path("./output_compare/metrics/sigma_z_no_dn.json"))
    ap.add_argument("--out_root", type=Path, default=Path("./output_compare"))
    ap.add_argument(
        "--tag_prefix",
        type=str,
        default="ggs_ex_trunc",
        help="Dirs are {out_root}/{tag_prefix}{m}/{scan} for m in T_GRID; "
        "legacy: trunc4→ggs_ex_trunc4, m=8 uses no_dn default extract.",
    )
    ap.add_argument("--scans", nargs="+", default=["scan24", "scan105", "scan122"])
    ap.add_argument("--out", type=Path, default=Path("./output_compare/metrics/g2_verdict.json"))
    args = ap.parse_args()

    sigma = json.loads(args.sigma_z.read_text(encoding="utf-8"))["sigma_z_by_scan"]

    # Resolve model dir for each (scan, m)
    def model_dir(scan: str, m: int) -> Path:
        if m == 8:
            # Default extract on frozen no_dn (original recon)
            return args.out_root / "ggs_ablate_no_dn" / scan
        if m == 4:
            # Prefer dedicated tag; fall back
            for tag in (f"ggs_ex_trunc{m}", f"ggs_ex_m{m}", f"{args.tag_prefix}{m}"):
                p = args.out_root / tag / scan
                if (p / "vis" / "results.json").exists():
                    return p
        for tag in (f"ggs_ex_m{m}", f"ggs_ex_trunc{m}", f"{args.tag_prefix}{m}"):
            p = args.out_root / tag / scan
            if (p / "vis" / "results.json").exists() or (p / "recon_post.ply").exists():
                return p
        return args.out_root / f"ggs_ex_m{m}" / scan

    table: dict[str, dict] = {}
    t_star: dict[str, float] = {}
    m_star: dict[str, int] = {}

    for scan in args.scans:
        rows = {}
        for m in T_GRID_M:
            t = m * VOXEL
            md = model_dir(scan, m)
            ov = load_overall(md)
            rows[m] = {"t": t, "overall": ov, "path": str(md)}
        # t* = argmin overall; ties → smaller t (protocol)
        valid = [(m, rows[m]["overall"]) for m in T_GRID_M if rows[m]["overall"] is not None]
        if not valid:
            table[scan] = {"error": "no overalls", "rows": rows}
            continue
        best_ov = min(ov for _, ov in valid)
        candidates = sorted(m for m, ov in valid if abs(ov - best_ov) < 1e-12)
        # also treat near-ties within SIGNIFICANT? Protocol: exact argmin, tie → smaller t
        # Use strict min; if multiple equal floats, smallest m
        ms = min(candidates)
        # If several within SIGNIFICANT of best, pick smallest m among those (favor tight)
        near = [m for m, ov in valid if ov <= best_ov + SIGNIFICANT]
        ms = min(near) if near else ms
        t_star[scan] = ms * VOXEL
        m_star[scan] = ms
        table[scan] = {
            "sigma_z": sigma.get(scan),
            "m_star": ms,
            "t_star": ms * VOXEL,
            "rows": rows,
        }

    # Overall variation across t (mean over scans of range)
    ranges = []
    for scan in args.scans:
        ovs = [table[scan]["rows"][m]["overall"] for m in T_GRID_M if table[scan]["rows"][m]["overall"] is not None]
        if len(ovs) >= 2:
            ranges.append(max(ovs) - min(ovs))
    mean_range = sum(ranges) / len(ranges) if ranges else 0.0

    stars = [m_star[s] for s in args.scans if s in m_star]
    sigs = [(s, sigma[s]) for s in args.scans if s in sigma and s in m_star]

    verdict = "M2-emp"
    reason = ""
    if len(set(stars)) == 1:
        verdict = "M2-const"
        reason = f"all t* equal (m={stars[0]}, t={stars[0]*VOXEL})"
    elif mean_range < SIGNIFICANT:
        verdict = "M2-emp"
        reason = f"overall variation across t < {SIGNIFICANT} (mean range={mean_range:.6e})"
    else:
        # Same order: sort scans by sigma_z ascending, t* should be non-decreasing
        order_sig = [s for s, _ in sorted(sigs, key=lambda x: x[1])]
        t_seq = [t_star[s] for s in order_sig]
        nondec = all(t_seq[i] <= t_seq[i + 1] + 1e-15 for i in range(len(t_seq) - 1))
        # allow one pair tied
        if nondec:
            verdict = "M2-adapt"
            reason = f"σ_z order {order_sig} → t* {[t_star[s] for s in order_sig]} non-decreasing"
        else:
            # check reverse
            noninc = all(t_seq[i] >= t_seq[i + 1] - 1e-15 for i in range(len(t_seq) - 1))
            if noninc:
                verdict = "M2-emp"
                reason = f"σ_z vs t* reverse order: {order_sig} → {t_seq}"
            else:
                verdict = "M2-emp"
                reason = f"σ_z vs t* unstructured: {order_sig} → {t_seq}"

    out = {
        "protocol": "RESEARCH_PROTOCOL §5 G2",
        "verdict": verdict,
        "reason": reason,
        "voxel": VOXEL,
        "significant": SIGNIFICANT,
        "mean_overall_range_across_t": mean_range,
        "m_star": m_star,
        "t_star": t_star,
        "sigma_z": {s: sigma.get(s) for s in args.scans},
        "table": table,
        "next": {
            "M2-const": "lock that t in Extract; proceed P3 (τ pilot)",
            "M2-adapt": "fit t=clip(k σ_z); hold-out scan prediction before Method",
            "M2-emp": "keep trunc4 as empirical default; demote M2 formula; still P3",
        }[verdict],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"VERDICT: {verdict}")
    print(f"reason: {reason}")
    for s in args.scans:
        if s in table and "m_star" in table[s]:
            print(
                f"  {s}: sigma_z={table[s].get('sigma_z')} "
                f"t*={table[s]['t_star']} (m={table[s]['m_star']})"
            )
            for m in T_GRID_M:
                ov = table[s]["rows"][m]["overall"]
                mark = " *" if m == table[s]["m_star"] else ""
                print(f"    m={m} t={m*VOXEL:.3f} overall={ov}{mark}")
    print(f"wrote {args.out}")
    print(f"NEXT: {out['next']}")


if __name__ == "__main__":
    main()
