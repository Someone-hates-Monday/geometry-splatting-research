#!/usr/bin/env python3
"""
Color an *aligned* reconstruction mesh by per-vertex distance to DTU GT (STL point cloud).

Prerequisite: run mesh eval once so `recon_aligned.ply` exists (same frame as DTU metrics):
  python evaluate_dtu_mesh.py -m <model_path> --iteration -1 --DTU <dtu_root>

This script does NOT re-run ICP; it only loads `recon_aligned.ply` and paints d2s-like error
(distance mesh vertex → nearest STL point). Colors are **continuous** in error magnitude
(matplotlib colormap on [0, vmax]). Use the same `--vmax` and `--colormap` across methods.

Triplet (baseline + M1-3 + M1-B) writes three PLY + one shared colorbar PNG.

Example:
  python scripts/export_mesh_vertex_error_dtu.py \\
    --mesh ./output_compare/ggs_baseline/scan24/recon_aligned.ply \\
    --dtu_dir ./dtu_eval/Offical_DTU_Dataset \\
    --scan 24 \\
    --out ./output_compare/ggs_baseline/scan24/vis/recon_aligned_d2s_vertex.ply

Pair (same color scale):
  python scripts/export_mesh_vertex_error_dtu.py \\
    --mesh_a ./output_compare/ggs_baseline/scan24/recon_aligned.ply \\
    --mesh_b ./output_compare/m1_ablate_03_full/scan24/recon_aligned.ply \\
    --dtu_dir ./dtu_eval/Offical_DTU_Dataset --scan 24 --vmax 6 \\
    --out_dir ./report_figures/scan24_err_compare

Triplet baseline + M1-3 + M1-B (pass in this order for meaningful summary keys):
  python scripts/export_mesh_vertex_error_dtu.py \\
    --mesh_a .../ggs_baseline/scan24/recon_aligned.ply \\
    --mesh_b .../m1_ablate_03_full/scan24/recon_aligned.ply \\
    --mesh_c .../m1_b_mesh_align/scan24/recon_aligned.ply \\
    --dtu_dir ./dtu_eval/Offical_DTU_Dataset --scan 24 --vmax 6 \\
    --out_dir ./report_figures/scan24_err_vertex_triple
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_stl_points(dtu_dir: Path, scan: int) -> np.ndarray:
    stl_path = dtu_dir / "Points" / "stl" / f"stl{scan:03d}_total.ply"
    if not stl_path.is_file():
        raise FileNotFoundError(f"Missing STL point cloud: {stl_path}")
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(stl_path))
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise RuntimeError(f"Empty STL ply: {stl_path}")
    return pts.astype(np.float64)


def _colors_from_distances(dists: np.ndarray, vmax: float, cmap_name: str) -> np.ndarray:
    """Map distances to RGB (N,3) in [0,1] using a continuous matplotlib colormap."""
    import matplotlib.pyplot as plt

    vmax = float(max(vmax, 1e-9))
    t = np.clip(np.asarray(dists, dtype=np.float64) / vmax, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(t)
    return rgba[:, :3].astype(np.float64)


def _save_colorbar_png(path: Path, cmap_name: str, vmin: float, vmax: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 1.15))
    fig.subplots_adjust(bottom=0.45, left=0.08, right=0.92, top=0.85)
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label("Distance to DTU STL (per vertex, same units as mesh eval)")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mesh_pair_label(mesh_path: Path) -> str:
    """Disambiguate .../METHOD/scanXX/recon_aligned.ply → METHOD_scanXX."""
    mesh_path = mesh_path.resolve()
    par = mesh_path.parent
    if par.name.lower().startswith("scan") and par.parent.name:
        return f"{par.parent.name}_{par.name}"
    return par.name.replace(" ", "_")


def _color_mesh_vertices_d2s(
    mesh_path: Path,
    stl_pts: np.ndarray,
    vmax: float,
    out_path: Path,
    cmap_name: str,
) -> dict:
    import open3d as o3d
    from sklearn.neighbors import NearestNeighbors

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Empty mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    nn.fit(stl_pts)
    dists, _ = nn.kneighbors(vertices, return_distance=True)
    dists = dists.reshape(-1)

    vmax_used = float(max(vmax, 1e-9))
    colors = _colors_from_distances(dists, vmax_used, cmap_name)

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out_path), mesh)

    meta = {
        "mesh": str(mesh_path),
        "out": str(out_path),
        "num_vertices": int(vertices.shape[0]),
        "vmax": vmax_used,
        "colormap": cmap_name,
        "color_mapping": "continuous linear [0, vmax] -> matplotlib colormap",
        "mean_d2s_vertex_to_stl": float(np.mean(dists)),
        "median_d2s_vertex_to_stl": float(np.median(dists)),
        "p95_d2s_vertex_to_stl": float(np.percentile(dists, 95)),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Vertex-color mesh by DTU STL distance (d2s proxy).")
    parser.add_argument("--dtu_dir", type=str, required=True, help="DTU eval root (with Points/stl/...)")
    parser.add_argument("--scan", type=int, required=True)
    parser.add_argument("--vmax", type=float, default=8.0, help="Upper end of color scale (distance saturates here; linear map from 0).")
    parser.add_argument(
        "--colormap",
        type=str,
        default="turbo",
        help="Matplotlib colormap name (continuous), e.g. turbo, viridis, plasma, inferno, cividis.",
    )
    parser.add_argument(
        "--no_colorbar",
        action="store_true",
        help="Do not write colorbar PNG (pair/triple default writes one shared bar).",
    )
    parser.add_argument(
        "--colorbar_out",
        type=str,
        default="",
        help="Single-mesh mode: path for colorbar PNG (optional).",
    )
    parser.add_argument("--mesh", type=str, default="", help="Single aligned mesh (recon_aligned.ply)")
    parser.add_argument("--out", type=str, default="", help="Output colored ply (single-mesh mode)")
    parser.add_argument("--mesh_a", type=str, default="", help="Aligned mesh A (pair/triple); use baseline first for triple")
    parser.add_argument("--mesh_b", type=str, default="", help="Aligned mesh B (pair/triple); use M1-3 for triple")
    parser.add_argument("--mesh_c", type=str, default="", help="Aligned mesh C (triple only); use M1-B")
    parser.add_argument(
        "--label_a",
        type=str,
        default="baseline",
        help="Triple summary key for mesh_a (default: baseline).",
    )
    parser.add_argument(
        "--label_b",
        type=str,
        default="M1-3",
        help="Triple summary key for mesh_b (default: M1-3).",
    )
    parser.add_argument(
        "--label_c",
        type=str,
        default="M1-B",
        help="Triple summary key for mesh_c (default: M1-B).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Pair/triple: output directory for colored meshes + JSON summary.",
    )
    args = parser.parse_args()

    dtu_dir = Path(args.dtu_dir).expanduser().resolve()
    stl_pts = _load_stl_points(dtu_dir, args.scan)

    try:
        import matplotlib.pyplot as plt

        plt.get_cmap(args.colormap)
    except Exception as e:
        parser.error(f"Invalid --colormap {args.colormap!r}: {e}")

    single = bool(args.mesh)
    triple = bool(args.mesh_a and args.mesh_b and args.mesh_c)
    pair = bool(args.mesh_a and args.mesh_b and not args.mesh_c)

    if single and (args.mesh_a or args.mesh_b or args.mesh_c):
        parser.error("Do not combine --mesh with --mesh_a/b/c.")
    if not single and not pair and not triple:
        parser.error("Use (--mesh + --out) OR (--mesh_a, --mesh_b, --out_dir) OR (--mesh_a,b,c, --out_dir).")
    if (pair or triple) and not args.out_dir:
        parser.error("--out_dir required for pair/triple mode.")

    if single:
        if not args.out:
            parser.error("--out required in single-mesh mode")
        meta = _color_mesh_vertices_d2s(
            Path(args.mesh), stl_pts, args.vmax, Path(args.out), args.colormap
        )
        if args.colorbar_out and not args.no_colorbar:
            _save_colorbar_png(Path(args.colorbar_out), args.colormap, 0.0, float(args.vmax))
        print(json.dumps(meta, indent=2))
        return

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ma = Path(args.mesh_a)
    mb = Path(args.mesh_b)
    la, lb = _mesh_pair_label(ma), _mesh_pair_label(mb)
    cmap_tag = args.colormap.replace(" ", "_")
    out_a = out_dir / f"{la}_d2s_vertex_{cmap_tag}_vmax{args.vmax:g}.ply"
    out_b = out_dir / f"{lb}_d2s_vertex_{cmap_tag}_vmax{args.vmax:g}.ply"
    meta_a = _color_mesh_vertices_d2s(ma, stl_pts, args.vmax, out_a, args.colormap)
    meta_b = _color_mesh_vertices_d2s(mb, stl_pts, args.vmax, out_b, args.colormap)

    if not args.no_colorbar:
        cb_path = out_dir / f"colorbar_{cmap_tag}_vmax{args.vmax:g}.png"
        _save_colorbar_png(cb_path, args.colormap, 0.0, float(args.vmax))

    if triple:
        mc = Path(args.mesh_c)
        lc = _mesh_pair_label(mc)
        out_c = out_dir / f"{lc}_d2s_vertex_{cmap_tag}_vmax{args.vmax:g}.ply"
        meta_c = _color_mesh_vertices_d2s(mc, stl_pts, args.vmax, out_c, args.colormap)
        mean_a = meta_a["mean_d2s_vertex_to_stl"]
        mean_b = meta_b["mean_d2s_vertex_to_stl"]
        mean_c = meta_c["mean_d2s_vertex_to_stl"]
        la, lb, lc_key = args.label_a, args.label_b, args.label_c
        summary = {
            "scan": args.scan,
            "dtu_dir": str(dtu_dir),
            "vmax": args.vmax,
            "colormap": args.colormap,
            "colorbar_png": str(out_dir / f"colorbar_{cmap_tag}_vmax{args.vmax:g}.png")
            if not args.no_colorbar
            else None,
            "order_note": (
                f"mesh_a={la}, mesh_b={lb}, mesh_c={lc_key} "
                "(vertex mean is tail-heavy; prefer median / official DTU mean_d2s)"
            ),
            la: meta_a,
            lb: meta_b,
            lc_key: meta_c,
            f"delta_mean_{la}_minus_{lb}": mean_a - mean_b,
            f"delta_mean_{la}_minus_{lc_key}": mean_a - mean_c,
            f"delta_mean_{lb}_minus_{lc_key}": mean_b - mean_c,
        }
        summary_path = out_dir / "compare_vertex_d2s_triplet_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary_path.read_text(encoding="utf-8"))
        return

    summary = {
        "scan": args.scan,
        "dtu_dir": str(dtu_dir),
        "vmax": args.vmax,
        "colormap": args.colormap,
        "colorbar_png": str(out_dir / f"colorbar_{cmap_tag}_vmax{args.vmax:g}.png")
        if not args.no_colorbar
        else None,
        "A": meta_a,
        "B": meta_b,
        "delta_mean_A_minus_B": meta_a["mean_d2s_vertex_to_stl"] - meta_b["mean_d2s_vertex_to_stl"],
    }
    summary_path = out_dir / "compare_vertex_d2s_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
