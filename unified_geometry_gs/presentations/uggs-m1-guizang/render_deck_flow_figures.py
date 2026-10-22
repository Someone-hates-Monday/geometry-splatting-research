# -*- coding: utf-8 -*-
"""
Generate PNG flowcharts (LaTeX-style math + boxes) for deck slides.
Uses ASCII/English labels so figures render on Windows without CJK fonts.

  py -3 unified_geometry_gs/presentations/uggs-m1-guizang/render_deck_flow_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "images"


def _rounded_box(ax, xy, w, h, text, fontsize=9, fc="#f4f4f2", ec="#222"):
    x, y = xy
    r = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(r)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111",
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#0b57d0", lw=1.4, shrinkA=0, shrinkB=0),
    )


def draw_row_baseline(ax_title: str, ax: plt.Axes, accent: str) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax.text(0.05, 2.05, ax_title, fontsize=11, fontweight="600", color=accent)

    ax.text(
        0.05,
        1.72,
        r"$L = L_{\mathrm{rgb}} + \lambda_{\mathrm{dn}}L_{\mathrm{dn}} + \lambda_{\mathrm{ncc}}L_{\mathrm{ncc}} + \lambda_{\mathrm{geo}}L_{\mathrm{geo}}$",
        fontsize=10,
        color="#333",
    )

    yb = 0.35
    h = 0.85
    w = 1.55
    xs = [0.35, 2.15, 3.95, 5.75, 7.55]
    labels = [
        "Multi-view\nRGB",
        "Pose\n(COLMAP)",
        "Diff.\nsplat",
        "Adam\nbackward",
        r"Gaussians" + "\n" + r"$\Theta$",
    ]
    for x, lab in zip(xs, labels, strict=True):
        _rounded_box(ax, (x, yb), w, h, lab, fontsize=8.5)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + w, yb + h * 0.5, xs[i + 1], yb + h * 0.5)

    ax.text(
        0.35,
        0.08,
        "Mesh: marching / Poisson  ->  triangle mesh  ->  evaluate_dtu_mesh  ->  JSON",
        fontsize=8,
        color="#555",
    )


def draw_row_sugar(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    ax.text(0.05, 2.05, "B · SuGaR-style post-process (off training loop)", fontsize=11, fontweight="600", color="#0b57d0")

    ax.text(
        0.05,
        1.72,
        r"surface objective: squash Gaussians + bind surface  $\Rightarrow$  mesh",
        fontsize=10,
        color="#333",
    )
    yb = 0.35
    h = 0.85
    w = 1.75
    xs = [0.35, 2.45, 4.55, 6.65]
    labels = [
        "Trained\n3DGS",
        "Density /\nellipsoid field",
        "Surface\nalignment",
        "Export\nmesh",
    ]
    for x, lab in zip(xs, labels, strict=True):
        _rounded_box(ax, (x, yb), w, h, lab, fontsize=8.5, fc="#eef5ff")
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + w, yb + h * 0.5, xs[i + 1], yb + h * 0.5)
    ax.text(
        0.35,
        0.08,
        "Optional chain with M1-trained GS: same eval scripts after export.",
        fontsize=8,
        color="#555",
    )


def draw_row_m1(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.35)
    ax.axis("off")
    ax.text(0.05, 2.18, "C · M1 in-loop regularizers (parallel to photometric path)", fontsize=11, fontweight="600", color="#111")

    ax.text(
        0.05,
        1.78,
        r"$L_{\mathrm{extra}}=\lambda_{\mathrm{occ}}L_{\mathrm{occ}}+\lambda_{\mathrm{doc}}L_{\mathrm{doc}}+\lambda_{\mathrm{rank}}L_{\mathrm{rank}}+\lambda_{\mathrm{thin}}L_{\mathrm{thin}}+\lambda_{\mathrm{ng}}L_{\mathrm{ng}}$",
        fontsize=9.2,
        color="#333",
    )

    yb = 0.38
    h = 0.82
    w = 1.35
    xs = [0.25, 1.85, 3.45, 5.05, 6.65, 8.25]
    labels = [
        r"screen" + "\n" + r"$\alpha,d$",
        "Occ",
        "Doc",
        "rank /\nthin",
        "M1-B\n(ng)",
        r"$\Sigma$ bp",
    ]
    for x, lab in zip(xs, labels, strict=True):
        fc = "#ffecec" if "\\mathrm{ng}" in lab else "#f4f4f2"
        _rounded_box(ax, (x, yb), w, h, lab, fontsize=8, fc=fc)
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + w, yb + h * 0.5, xs[i + 1], yb + h * 0.5)

    ax.text(
        0.25,
        0.06,
        "Same backward as L_rgb / L_ncc / L_geo; inference path unchanged.",
        fontsize=8,
        color="#555",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "figure.dpi": 150,
        }
    )

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(3, 1, hspace=0.22, left=0.04, right=0.98, top=0.96, bottom=0.04)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    draw_row_baseline("A · Baseline 3DGS (control)", ax0, "#333")
    draw_row_sugar(ax1)
    draw_row_m1(ax2)

    out = OUT_DIR / "deck_flow_models_triptych.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()
