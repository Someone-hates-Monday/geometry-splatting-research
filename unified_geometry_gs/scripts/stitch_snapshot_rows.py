#!/usr/bin/env python3
"""
Horizontally stitch capture_mesh_snapshots outputs so each PNG row is:
  [label_a_view] [label_b_view] ...

Expects files: in_dir/{label}_{view_stem}.png (orbit9 → many view stems).

Example:
  python scripts/stitch_snapshot_rows.py \\
    --in_dir ./report_figures_edge_compare/scan65/orbit9_raw \\
    --labels baseline edge_dn pm_edge \\
    --out_dir ./report_figures_edge_compare/scan65/orbit9_rows
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="Order left→right")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args()

    raw = Path(args.in_dir).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    labels = [lb.replace(" ", "_") for lb in args.labels]
    out.mkdir(parents=True, exist_ok=True)

    ref = labels[0]
    views: list[str] = []
    for p in sorted(raw.glob(f"{ref}_*.png")):
        views.append(p.stem[len(ref) + 1 :])

    if not views:
        print(f"No PNGs matching {raw}/{ref}_*.png", flush=True)
        return

    for view in views:
        imgs: list[Image.Image] = []
        skip = False
        for lb in labels:
            fp = raw / f"{lb}_{view}.png"
            if not fp.is_file():
                print(f"SKIP row {view}: missing {fp.name}", flush=True)
                skip = True
                break
            imgs.append(Image.open(fp).convert("RGB"))
        if skip:
            continue
        w = sum(im.width for im in imgs)
        h = max(im.height for im in imgs)
        canvas = Image.new("RGB", (w, h), (255, 255, 255))
        x = 0
        for im in imgs:
            y = (h - im.height) // 2
            canvas.paste(im, (x, y))
            x += im.width
        outp = out / f"row_{view}.png"
        canvas.save(outp)
        print(f"Wrote {outp}", flush=True)

    # One composite: stack rows (limit height — orbit9 has 9 rows)
    rows = sorted(out.glob("row_*.png"))
    if len(rows) <= 1:
        return
    ims = [Image.open(p).convert("RGB") for p in rows]
    tw = max(im.width for im in ims)
    th = sum(im.height for im in ims)
    mega = Image.new("RGB", (tw, th), (255, 255, 255))
    y = 0
    for im in ims:
        x = (tw - im.width) // 2
        mega.paste(im, (x, y))
        y += im.height
    mega_path = out / "_all_views_stacked.png"
    mega.save(mega_path)
    print(f"Wrote {mega_path} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
