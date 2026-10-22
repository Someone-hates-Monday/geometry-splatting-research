#!/usr/bin/env python3
"""
Check DTU eval data for mesh quality comparison and print setup instructions.

Run from repo root:
  python scripts/check_dtu_eval_setup.py [--scan 105]
"""
from __future__ import annotations

import argparse
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_DTU_ROOT = _PROJECT_ROOT / "dtu_eval" / "Offical_DTU_Dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check DTU eval data setup for mesh comparison")
    parser.add_argument("--scan", type=int, default=105, help="DTU scan id (e.g. 105)")
    parser.add_argument("--dtu_root", type=str, default=str(_DEFAULT_DTU_ROOT), help="DTU eval root")
    args = parser.parse_args()

    root = Path(args.dtu_root)
    scan = args.scan

    print("DTU mesh evaluation data check")
    print("=" * 60)
    print(f"Scan: {scan}")
    print(f"Expected root: {root.resolve()}")
    print()

    missing = []
    # Calibration
    cal = root / "Calibration"
    if not cal.is_dir():
        missing.append(f"  Missing folder: {cal}")
    else:
        pos1 = cal / "cal18" / "pos_001.txt"
        pos49 = cal / "cal49" / "pos_001.txt"
        if not pos1.is_file() and not pos49.is_file():
            found = False
            for sub in cal.iterdir():
                if sub.is_dir() and (sub / "pos_001.txt").is_file():
                    found = True
                    break
            if not found:
                missing.append(f"  Missing: Calibration/cal18/pos_001.txt (or cal49/pos_001.txt) under {root}")
        else:
            print("  [OK] Calibration (cal18 or cal49)")
    if not (root / "ObsMask" / f"ObsMask{scan}_10.mat").is_file():
        missing.append(f"  Missing file: ObsMask/ObsMask{scan}_10.mat")
    else:
        print(f"  [OK] ObsMask/ObsMask{scan}_10.mat")
    if not (root / "ObsMask" / f"Plane{scan}.mat").is_file():
        missing.append(f"  Missing file: ObsMask/Plane{scan}.mat")
    else:
        print(f"  [OK] ObsMask/Plane{scan}.mat")
    stl = root / "Points" / "stl" / f"stl{scan:03d}_total.ply"
    if not stl.is_file():
        missing.append(f"  Missing file: Points/stl/stl{scan:03d}_total.ply")
    else:
        print(f"  [OK] Points/stl/stl{scan:03d}_total.ply")

    if missing:
        print()
        print("Missing items:")
        for m in missing:
            print(m)
        print()
        print("How to fix:")
        print("  1) Download from DTU Robot Image Dataset:")
        print("     https://roboimagedata.compute.dtu.dk/?page_id=36")
        print()
        print("  2) Get these files:")
        print("     - SampleSet.zip  -> contains Calibration/ and ObsMask/")
        print("     - Points.zip     -> extract to Points/ (contains stl/stlXXX_total.ply)")
        print()
        print("  3) Place them so the structure is:")
        print(f"     {root}/")
        print("       Calibration/cal18/pos_001.txt ... pos_064.txt")
        print(f"       ObsMask/ObsMask{scan}_10.mat, Plane{scan}.mat")
        print(f"       Points/stl/stl{scan:03d}_total.ply")
        print()
        print("  4) See INSTALL.md for details.")
    else:
        print()
        print("All required files present. Mesh quality comparison will run.")
        print("Run: python scripts/compare_models.py --baseline <path> --optimized <path> --out <out.json>")


if __name__ == "__main__":
    main()
