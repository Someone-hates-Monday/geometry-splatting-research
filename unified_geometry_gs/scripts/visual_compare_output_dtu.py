#!/usr/bin/env python3
"""
One-shot visual comparison for ./output/dtu/<scan> vs <scan>_<variant>.

Produces under ./report_figures_edge_compare/<scan>/:
  - orbit9_raw/          PNGs per method (capture_mesh_snapshots orbit9)
  - orbit9_rows/         Same viewpoint stitched left→right for quick difference spotting
  - vertex_err_d2s/      Vertex-colored aligned meshes (distance to DTU STL) + colorbar + JSON

Windows / corners are rare in global metrics; **vertex error maps** and **zoomed orbit9**
make local geometry differences visible.

Usage (project root, conda env with torch/open3d/matplotlib):
  python scripts/visual_compare_output_dtu.py --scans scan65 scan24 \\
    --variants baseline edge_dn pm_edge --dtu_dir ./dtu_eval/Offical_DTU_Dataset

  # Only stitching (snapshots already in orbit9_raw/)
  python scripts/visual_compare_output_dtu.py --scans scan65 --stitch_only

Notes:
  - Requires recon_post.ply per run folder; runs evaluate_dtu_mesh if recon_aligned.ply missing.
  - Uses recon_post for orbit snapshots (camera-frame mesh); recon_aligned for STL vertex coloring.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent


def _norm_scan(s: str) -> str:
    s = s.strip().lower()
    if s.isdigit():
        return f"scan{int(s)}"
    if re.match(r"^scan\d+$", s):
        return s
    raise ValueError(f"Bad scan name: {s!r} (use scan65 or 65)")


def _model_dir(output_dt: Path, scan: str, variant: str) -> Path:
    if variant == "baseline":
        return output_dt / scan
    return output_dt / f"{scan}_{variant}"


def _ensure_aligned(py: str, model_dir: Path, dtu_dir: Path) -> bool:
    post = model_dir / "recon_post.ply"
    aligned = model_dir / "recon_aligned.ply"
    if not post.is_file():
        return False
    if aligned.is_file():
        return True
    cmd = [
        py,
        str(_PROJECT_ROOT / "evaluate_dtu_mesh.py"),
        "-m",
        str(model_dir),
        "--iteration",
        "-1",
        "--DTU",
        str(dtu_dir),
    ]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    return r.returncode == 0 and aligned.is_file()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scans",
        nargs="+",
        required=True,
        help="e.g. scan65 65 scan24",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "edge_dn", "pm_edge"],
        help="baseline = ./output/dtu/<scan>; others = <scan>_<variant>",
    )
    ap.add_argument(
        "--output_dt",
        type=str,
        default="./output/dtu",
        help="Root containing scan folders",
    )
    ap.add_argument("--dtu_dir", type=str, default="./dtu_eval/Offical_DTU_Dataset")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument(
        "--skip_eval",
        action="store_true",
        help="Do not run evaluate_dtu_mesh (aligned ply must exist for vertex export)",
    )
    ap.add_argument(
        "--skip_snapshots",
        action="store_true",
        help="Skip capture_mesh_snapshots",
    )
    ap.add_argument(
        "--skip_vertex",
        action="store_true",
        help="Skip export_mesh_vertex_error_dtu",
    )
    ap.add_argument(
        "--stitch_only",
        action="store_true",
        help="Only run stitch_snapshot_rows on existing orbit9_raw/",
    )
    ap.add_argument("--vmax_vertex", type=float, default=6.0)
    ap.add_argument("--colormap_vertex", type=str, default="turbo")
    ap.add_argument("--snapshot_width", type=int, default=1280)
    ap.add_argument("--snapshot_height", type=int, default=960)
    args = ap.parse_args()

    output_dt = Path(args.output_dt).expanduser().resolve()
    dtu_dir = Path(args.dtu_dir).expanduser().resolve()
    py = args.python

    scans = [_norm_scan(s) for s in args.scans]

    for scan in scans:
        scan_num = int(scan.replace("scan", ""))
        base_out = _PROJECT_ROOT / "report_figures_edge_compare" / scan
        raw_snap = base_out / "orbit9_raw"
        rows_snap = base_out / "orbit9_rows"
        vert_out = base_out / "vertex_err_d2s"

        variants = list(args.variants)
        entries: list[str] = []
        aligned_paths: list[Path] = []

        for v in variants:
            md = _model_dir(output_dt, scan, v)
            label = v.replace(" ", "_")
            if not (md / "recon_post.ply").is_file():
                print(f"[{scan}] skip variant {v}: no recon_post at {md}", flush=True)
                continue
            entries.append(f"{label}:{md / 'recon_post.ply'}")
            if not args.skip_eval and not args.stitch_only:
                _ensure_aligned(py, md, dtu_dir)
            al = md / "recon_aligned.ply"
            if al.is_file():
                aligned_paths.append(al)

        if args.stitch_only:
            labs = [v.replace(" ", "_") for v in variants]
            subprocess.run(
                [
                    py,
                    str(_SCRIPT_DIR / "stitch_snapshot_rows.py"),
                    "--in_dir",
                    str(raw_snap),
                    "--labels",
                    *labs,
                    "--out_dir",
                    str(rows_snap),
                ],
                cwd=str(_PROJECT_ROOT),
                check=False,
            )
            continue

        if len(entries) < 2:
            print(f"[{scan}] need ≥2 valid variants with recon_post; got {entries}", flush=True)
            continue

        raw_snap.mkdir(parents=True, exist_ok=True)

        if not args.skip_snapshots:
            cmd = [
                py,
                str(_SCRIPT_DIR / "capture_mesh_snapshots.py"),
                "--backend",
                "matplotlib",
                "--mat_preset",
                "orbit9",
                "--out_dir",
                str(raw_snap),
                "--width",
                str(args.snapshot_width),
                "--height",
                str(args.snapshot_height),
                "--simplify",
                "200000",
                *entries,
            ]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(_PROJECT_ROOT), check=True)

            labs = [e.split(":", 1)[0] for e in entries]
            subprocess.run(
                [
                    py,
                    str(_SCRIPT_DIR / "stitch_snapshot_rows.py"),
                    "--in_dir",
                    str(raw_snap),
                    "--labels",
                    *labs,
                    "--out_dir",
                    str(rows_snap),
                ],
                cwd=str(_PROJECT_ROOT),
                check=True,
            )

        if args.skip_vertex or len(aligned_paths) < 2:
            if len(aligned_paths) < 2:
                print(f"[{scan}] vertex export needs ≥2 recon_aligned.ply", flush=True)
            continue

        vert_out.mkdir(parents=True, exist_ok=True)
        exp = _SCRIPT_DIR / "export_mesh_vertex_error_dtu.py"
        # Prefer baseline first if present
        order_keys = [v.replace(" ", "_") for v in variants]
        label_to_path = {}
        for v in variants:
            md = _model_dir(output_dt, scan, v)
            al = md / "recon_aligned.ply"
            if al.is_file():
                label_to_path[v.replace(" ", "_")] = al

        ordered = [label_to_path[k] for k in order_keys if k in label_to_path]
        if len(ordered) >= 3:
            cmd_v = [
                py,
                str(exp),
                "--dtu_dir",
                str(dtu_dir),
                "--scan",
                str(scan_num),
                "--vmax",
                str(args.vmax_vertex),
                "--colormap",
                args.colormap_vertex,
                "--mesh_a",
                str(ordered[0]),
                "--mesh_b",
                str(ordered[1]),
                "--mesh_c",
                str(ordered[2]),
                "--out_dir",
                str(vert_out),
            ]
        else:
            cmd_v = [
                py,
                str(exp),
                "--dtu_dir",
                str(dtu_dir),
                "--scan",
                str(scan_num),
                "--vmax",
                str(args.vmax_vertex),
                "--colormap",
                args.colormap_vertex,
                "--mesh_a",
                str(ordered[0]),
                "--mesh_b",
                str(ordered[1]),
                "--out_dir",
                str(vert_out),
            ]
        print(" ".join(cmd_v), flush=True)
        subprocess.run(cmd_v, cwd=str(_PROJECT_ROOT), check=True)

        print(f"\n[{scan}] Done.\n  Rows: {rows_snap}\n  Vertex PLY: {vert_out}\n", flush=True)

    print(
        "Open: report_figures_edge_compare/<scan>/orbit9_rows/row_*.png "
        "and _all_views_stacked.png\n"
        "MeshLab / CloudCompare: vertex_err_d2s/*_d2s_vertex_*.ply with colorbar PNG.",
        flush=True,
    )


if __name__ == "__main__":
    main()
