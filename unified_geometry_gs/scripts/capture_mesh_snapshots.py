#!/usr/bin/env python3
"""
Offline snapshots of triangle meshes for reports.

Backends (WSL-friendly order):
  1) open3d.visualization.rendering.OffscreenRenderer — no GLFW window (preferred on WSL).
  2) matplotlib + mesh triangles — pure software, slow but works without GL.

Example:
  python scripts/capture_mesh_snapshots.py --out_dir ./report_figures/scan24 \\
    baseline:./output_compare/ggs_baseline/scan24/recon_post.ply \\
    M1-3:./output_compare/m1_ablate_03_full/scan24/recon_post.ply \\
    M1-B:./output_compare/m1_b_mesh_align/scan24/recon_post.ply

If offscreen still fails, force matplotlib:
  python scripts/capture_mesh_snapshots.py --backend matplotlib ...

Orientation still wrong? Try --mat_flip none | y | xz (COLMAP / scenes vary).
Detail packs (9 PNGs per mesh; strong zoom by default — lower --mat_zoom if corners clip):
  --mat_preset grid9    3×3 (elevation × azimuth)
  --mat_preset orbit9   8 azimuth steps + 1 top (good “full orbit” contact sheet)
  Omit --mat_zoom / --mat_padding to use preset defaults; pass them to override.

Quick 3×3 poster (ImageMagick): montage *_g9_*.png -tile 3x3 -geometry +4+4 out.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _load_mesh(ply_path: str, simplify_triangles: int | None):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(ply_path)
    if len(mesh.vertices) == 0:
        raise RuntimeError(f"Empty mesh: {ply_path}")
    mesh.compute_vertex_normals()
    if simplify_triangles is not None and simplify_triangles > 0:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(simplify_triangles))
    return mesh


# (name, elev_deg, azim_deg) — matplotlib Axes3D.view_init (degrees); offsets applied at runtime
_MATPLOTLIB_VIEWS_BASIC = [
    ("front", 14.0, -90.0),
    ("diag", 23.0, -124.0),
    ("top", 90.0, -90.0),
]


def _matplotlib_views_for_preset(preset: str) -> list[tuple[str, float, float]]:
    """Return named views: (stem, elev, azim) for matplotlib.view_init."""
    p = preset.lower().strip()
    if p == "basic":
        return list(_MATPLOTLIB_VIEWS_BASIC)
    if p == "grid9":
        # 3 rows: lower / mid / higher pitch; 3 cols: yaw sweep (around typical DTU façade)
        elevs = [4.5, 13.5, 25.0]
        azims = [-136.0, -90.0, -44.0]
        views: list[tuple[str, float, float]] = []
        for ri, e in enumerate(elevs):
            for ci, a in enumerate(azims):
                views.append((f"g9_r{ri}c{ci}_e{int(e)}_a{int(a)}", float(e), float(a)))
        assert len(views) == 9
        return views
    if p == "orbit9":
        views_o: list[tuple[str, float, float]] = []
        for k in range(8):
            a = -180.0 + k * 45.0
            if a == 0.0:
                tag = "a0"
            elif a < 0:
                tag = f"a_m{int(abs(a))}"
            else:
                tag = f"a_p{int(a)}"
            views_o.append((f"orbit_{tag}", 13.0, a))
        views_o.append(("orbit_top", 86.5, -90.0))
        assert len(views_o) == 9
        return views_o
    raise ValueError(f"Unknown mat_preset: {preset!r} (use basic, grid9, orbit9)")


def _matplotlib_zoom_pad_defaults(preset: str) -> tuple[float, float]:
    """(mat_zoom, mat_padding) when user does not override."""
    p = preset.lower().strip()
    if p == "basic":
        return (1.18, 0.004)
    if p in ("grid9", "orbit9"):
        # Strong close-up for picking details (~2× vs older grid default ~2.9); reduce if mesh clipped
        return (5.55, 0.0005)
    raise ValueError(preset)


def _apply_vertex_axis_flip(vertices: np.ndarray, flip_spec: str) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64).copy()
    spec = flip_spec.lower().replace(",", "").replace(" ", "")
    if spec in ("", "none", "no"):
        return v
    for ch in spec:
        if ch == "x":
            v[:, 0] *= -1.0
        elif ch == "y":
            v[:, 1] *= -1.0
        elif ch == "z":
            v[:, 2] *= -1.0
        else:
            raise ValueError(f"Unknown axis in --mat_flip: {ch!r} (use x,y,z,none)")
    return v


def capture_views_matplotlib(
    mesh,
    out_prefix: str,
    width: int,
    height: int,
    *,
    views: list[tuple[str, float, float]],
    padding: float,
    flip_spec: str,
    elev_offset: float,
    azim_offset: float,
    zoom_fill: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = _apply_vertex_axis_flip(np.asarray(mesh.vertices), flip_spec)
    triangles = np.asarray(mesh.triangles)
    polys = vertices[triangles]

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    ranges = np.maximum(bbox_max - bbox_min, 1e-9)
    pad = float(padding) * ranges
    lo = bbox_min - pad
    hi = bbox_max + pad

    zf = max(float(zoom_fill), 1.0)
    c = (lo + hi) * 0.5
    half = (hi - lo) * 0.5 / zf
    lo_v = c - half
    hi_v = c + half
    box_r = hi_v - lo_v

    fig_dpi = max(int(dpi), 72)

    for name, elev, azim in views:
        fig = plt.figure(
            figsize=(width / float(fig_dpi), height / float(fig_dpi)),
            dpi=fig_dpi,
            facecolor="white",
        )
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        coll = Poly3DCollection(
            polys,
            linewidths=0.04,
            alpha=1.0,
            facecolor=(0.72, 0.74, 0.78),
            edgecolor=(0.42, 0.44, 0.48),
        )
        ax.add_collection3d(coll)

        ax.set_xlim(lo_v[0], hi_v[0])
        ax.set_ylim(lo_v[1], hi_v[1])
        ax.set_zlim(lo_v[2], hi_v[2])
        try:
            ax.set_box_aspect((box_r[0], box_r[1], box_r[2]))
        except Exception:
            pass
        ax.view_init(elev=elev + float(elev_offset), azim=azim + float(azim_offset))
        ax.set_axis_off()
        out_path = f"{out_prefix}_{name}.png"
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0, facecolor="white")
        plt.close(fig)
        print(f"Wrote {out_path} (matplotlib)")


def capture_views_offscreen(
    mesh,
    out_prefix: str,
    width: int,
    height: int,
) -> None:
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    extent = float(np.linalg.norm(bbox.get_extent())) + 1e-9

    views = [
        ("front", np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0]), 1.05),
        ("diag", np.array([1.0, 0.55, 0.75]), np.array([0.0, -1.0, 0.0]), 1.2),
        ("top", np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), 1.15),
    ]

    for name, forward_world, up_world, dist_scale in views:
        fw = forward_world / (np.linalg.norm(forward_world) + 1e-12)
        up = up_world / (np.linalg.norm(up_world) + 1e-12)
        dist = extent * dist_scale
        eye = center + fw * dist

        renderer = rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("mesh", mesh, mat)
        renderer.setup_camera(60.0, center.tolist(), eye.tolist(), up.tolist())
        img = renderer.render_to_image()
        out_path = f"{out_prefix}_{name}.png"
        o3d.io.write_image(out_path, img)
        print(f"Wrote {out_path} (offscreen)")


def capture_views_glfw(
    mesh,
    out_prefix: str,
    width: int,
    height: int,
) -> None:
    """Legacy GLFW path — often breaks on WSL without working GL."""
    import open3d as o3d

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    vis = o3d.visualization.Visualizer()
    if not vis.create_window(width=width, height=height, visible=False):
        raise RuntimeError("create_window failed")
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    ctr.set_lookat(center)

    views = [
        ("front", np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0]), 0.65),
        ("diag", np.array([-1.0, -0.45, -0.65]), np.array([0.0, -1.0, 0.0]), 0.55),
        ("top", np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.55),
    ]

    for name, front, up, zoom in views:
        f = front / (np.linalg.norm(front) + 1e-12)
        u = up / (np.linalg.norm(up) + 1e-12)
        ctr.set_front(f.tolist())
        ctr.set_up(u.tolist())
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        out_path = f"{out_prefix}_{name}.png"
        vis.capture_screen_image(out_path, do_render=True)
        print(f"Wrote {out_path} (glfw)")

    vis.destroy_window()


def capture_views(
    ply_path: str,
    out_prefix: str,
    width: int,
    height: int,
    simplify_triangles: int | None,
    backend: str,
    *,
    mat_views: list[tuple[str, float, float]],
    mat_padding: float,
    mat_flip: str,
    mat_elev_offset: float,
    mat_azim_offset: float,
    mat_zoom: float,
    mat_dpi: int,
) -> None:
    mesh = _load_mesh(ply_path, simplify_triangles)

    if backend == "matplotlib":
        capture_views_matplotlib(
            mesh,
            out_prefix,
            width,
            height,
            views=mat_views,
            padding=mat_padding,
            flip_spec=mat_flip,
            elev_offset=mat_elev_offset,
            azim_offset=mat_azim_offset,
            zoom_fill=mat_zoom,
            dpi=mat_dpi,
        )
        return
    if backend == "glfw":
        capture_views_glfw(mesh, out_prefix, width, height)
        return
    if backend == "offscreen":
        capture_views_offscreen(mesh, out_prefix, width, height)
        return

    # auto
    try:
        capture_views_offscreen(mesh, out_prefix, width, height)
        return
    except Exception as e:
        print(f"[capture_mesh_snapshots] offscreen failed ({e}); trying matplotlib...", file=sys.stderr)
    try:
        capture_views_matplotlib(
            mesh,
            out_prefix,
            width,
            height,
            views=mat_views,
            padding=mat_padding,
            flip_spec=mat_flip,
            elev_offset=mat_elev_offset,
            azim_offset=mat_azim_offset,
            zoom_fill=mat_zoom,
            dpi=mat_dpi,
        )
        return
    except Exception as e:
        print(f"[capture_mesh_snapshots] matplotlib failed ({e}); trying glfw...", file=sys.stderr)
    capture_views_glfw(mesh, out_prefix, width, height)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture mesh PNGs from fixed viewpoints.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for PNG outputs")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument(
        "--simplify",
        type=int,
        default=250_000,
        help="Quadric decimation target triangles (0 = disable).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("auto", "offscreen", "matplotlib", "glfw"),
        default="auto",
        help="Rendering backend; use matplotlib on broken GL/WSL.",
    )
    parser.add_argument(
        "--mat_preset",
        type=str,
        choices=("basic", "grid9", "orbit9"),
        default="basic",
        help="Matplotlib: basic (3 PNG), grid9 (9 PNG), orbit9 (8 orbit + top).",
    )
    parser.add_argument(
        "--mat_padding",
        type=float,
        default=None,
        help="Matplotlib: bbox padding fraction per axis; omit = preset default.",
    )
    parser.add_argument(
        "--mat_zoom",
        type=float,
        default=None,
        help="Matplotlib: crop toward center (>=1); omit = preset default (~5.5 for grid9/orbit9).",
    )
    parser.add_argument(
        "--mat_elev_offset",
        type=float,
        default=-7.0,
        help="Matplotlib: added to view elev (negative ≈ look less from above / 俯视少一点).",
    )
    parser.add_argument(
        "--mat_azim_offset",
        type=float,
        default=9.0,
        help="Matplotlib: added to view azim (yaw / 绕竖直轴转正，反向则用负数).",
    )
    parser.add_argument(
        "--mat_dpi",
        type=int,
        default=120,
        help="Matplotlib figure DPI (pixel size ≈ width/height at this dpi).",
    )
    parser.add_argument(
        "--mat_flip",
        type=str,
        default="z",
        help="Matplotlib: negate axes to fix upside-down (e.g. z, y, xz, none). Default z.",
    )
    parser.add_argument(
        "entries",
        nargs="+",
        help="Pairs label:path/to/recon_post.ply",
    )
    args = parser.parse_args()

    mat_views = _matplotlib_views_for_preset(args.mat_preset)
    z_def, p_def = _matplotlib_zoom_pad_defaults(args.mat_preset)
    mat_zoom = args.mat_zoom if args.mat_zoom is not None else z_def
    mat_padding = args.mat_padding if args.mat_padding is not None else p_def

    os.makedirs(args.out_dir, exist_ok=True)
    simplify = None if args.simplify == 0 else args.simplify

    for raw in args.entries:
        if ":" not in raw:
            print(f"Skip invalid entry (need label:path): {raw}", file=sys.stderr)
            continue
        label, path = raw.split(":", 1)
        label = label.strip().replace(" ", "_")
        path = path.strip()
        if not os.path.isfile(path):
            print(f"Missing file [{label}]: {path}", file=sys.stderr)
            continue
        prefix = os.path.join(args.out_dir, label)
        capture_views(
            path,
            prefix,
            args.width,
            args.height,
            simplify,
            args.backend,
            mat_views=mat_views,
            mat_padding=mat_padding,
            mat_flip=args.mat_flip,
            mat_elev_offset=args.mat_elev_offset,
            mat_azim_offset=args.mat_azim_offset,
            mat_zoom=mat_zoom,
            mat_dpi=args.mat_dpi,
        )


if __name__ == "__main__":
    main()
