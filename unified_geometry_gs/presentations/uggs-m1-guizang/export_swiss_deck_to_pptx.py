# -*- coding: utf-8 -*-
"""
Export a guizang Swiss single-file HTML deck to:

1) **PPTX** — one slide per HTML page, full-bleed PNG after slide transition + Motion One
   recipes have had time to finish. This does **not** recreate PowerPoint shape animations;
   each slide is a **raster snapshot of the final frame**.

2) **Optional WebM** — Playwright screen recording for the whole run: **intra-slide motion is
   preserved** in the video file (embed in PPT or play separately).

Requires: ``pip install playwright python-pptx`` (Chromium optional: script prefers **system Edge / Chrome** via Playwright ``channel=``).

Regenerate index.html after template updates so `window.__exportGoToSlide` exists
(`py -3 build_deck.py` in this directory).
"""
from __future__ import annotations

import argparse
import http.server
import shutil
import socketserver
import tempfile
import threading
from io import BytesIO
from pathlib import Path


def _blank_layout(prs):
    for layout in prs.slide_layouts:
        name = (layout.name or "").lower()
        if "blank" in name:
            return layout
    # default wide theme: index 6 is often blank
    try:
        return prs.slide_layouts[6]
    except IndexError:
        return prs.slide_layouts[0]


def _serve_dir(directory: Path, port: int) -> tuple[socketserver.TCPServer, threading.Thread]:
    root = str(directory.resolve())

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=root, **kwargs)

    class ReuseAddr(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = ReuseAddr(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def _launch_chromium(p, *, headless: bool = True):
    """Prefer system Edge/Chrome so `playwright install chromium` is not required."""
    for channel in ("msedge", "chrome"):
        try:
            browser = p.chromium.launch(headless=headless, channel=channel)
            print(f"Using Chromium channel: {channel}", flush=True)
            return browser
        except Exception:
            continue
    browser = p.chromium.launch(headless=headless)
    print("Using Playwright-bundled Chromium (run: playwright install chromium if this fails)", flush=True)
    return browser


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="HTML Swiss deck → PPTX (+ optional WebM)")
    ap.add_argument(
        "--html-dir",
        type=Path,
        default=here,
        help="Directory containing index.html and assets (default: script dir)",
    )
    ap.add_argument("--url-path", default="/index.html", help="Path on local server")
    ap.add_argument("--port", type=int, default=8769)
    ap.add_argument("--out", type=Path, default=here / "deck_export.pptx")
    ap.add_argument("--video-out", type=Path, default=None, help="Optional path for full-session WebM (animations kept)")
    ap.add_argument("--viewport", default="1920x1080", help="WxH, e.g. 1920x1080 (16:9)")
    ap.add_argument(
        "--wait-ms",
        type=int,
        default=3400,
        help="Sleep after each __exportGoToSlide (ms); increase if formulas/images still settling",
    )
    ap.add_argument("--skip-pptx", action="store_true", help="Only record video")
    ap.add_argument("--skip-video", action="store_true", help="Only build PPTX")
    args = ap.parse_args()

    if args.skip_pptx and not args.video_out:
        raise SystemExit("--skip-pptx requires --video-out (nothing else would be produced).")

    html_dir: Path = args.html_dir.resolve()
    index = html_dir / "index.html"
    if not index.is_file():
        raise SystemExit(f"Missing {index}")

    w_s, h_s = args.viewport.lower().split("x", 1)
    vw, vh = int(w_s), int(h_s)

    httpd, _ = _serve_dir(html_dir, args.port)
    base = f"http://127.0.0.1:{args.port}{args.url_path}"

    pngs: list[bytes] = []
    video_tmp: str | None = None
    want_video = bool(args.video_out) and not args.skip_video

    try:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:  # pragma: no cover
            raise SystemExit("Missing playwright. Install: pip install playwright && playwright install chromium") from e

        with sync_playwright() as p:
            ctx_kw: dict = {
                "viewport": {"width": vw, "height": vh},
                "device_scale_factor": 1,
            }
            if want_video:
                video_tmp = tempfile.mkdtemp(prefix="swiss-deck-rec-")
                ctx_kw["record_video_dir"] = video_tmp
                ctx_kw["record_video_size"] = {"width": vw, "height": vh}

            browser = _launch_chromium(p, headless=True)
            context = browser.new_context(**ctx_kw)
            page = context.new_page()
            page.goto(base, wait_until="load", timeout=120_000)
            page.wait_for_timeout(800)
            try:
                page.wait_for_function(
                    "() => !window.MathJax || window.MathJax.startup === undefined || window.MathJax.typesetPromise",
                    timeout=30_000,
                )
            except Exception:
                pass
            page.wait_for_timeout(400)

            n = page.evaluate("() => document.querySelectorAll('#deck .slide').length")
            if not isinstance(n, int) or n < 1:
                context.close()
                browser.close()
                raise SystemExit("Could not count #deck .slide")

            has_hook = page.evaluate("() => typeof window.__exportGoToSlide === 'function'")
            if not has_hook:
                context.close()
                browser.close()
                raise SystemExit(
                    "window.__exportGoToSlide missing. Rebuild index from template "
                    "(tools/guizang-ppt-skill/assets/template-swiss.html) and run py -3 build_deck.py"
                )

            for i in range(n):
                page.evaluate(f"window.__exportGoToSlide({i})")
                page.wait_for_timeout(args.wait_ms)
                if not args.skip_pptx:
                    pngs.append(page.screenshot(type="png", full_page=False))

            context.close()
            browser.close()

            if want_video and video_tmp:
                outs = list(Path(video_tmp).rglob("*.webm"))
                dest = args.video_out.resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)
                if outs:
                    shutil.move(str(outs[0]), str(dest))
                    print("Wrote video", dest)
                else:
                    import sys

                    print("Warning: no WebM produced under", video_tmp, file=sys.stderr)

    finally:
        if want_video and video_tmp:
            shutil.rmtree(video_tmp, ignore_errors=True)
        httpd.shutdown()

    if not args.skip_pptx:
        try:
            from pptx import Presentation
            from pptx.util import Inches
        except ImportError as e:  # pragma: no cover
            raise SystemExit("Missing python-pptx. Install: pip install python-pptx") from e

        if not pngs:
            raise SystemExit("No PNG captures (enable PPTX export or check --skip-pptx).")

        prs = Presentation()
        prs.slide_width = Inches(13.333333333333334)
        prs.slide_height = Inches(7.5)
        layout = _blank_layout(prs)
        for buf in pngs:
            slide = prs.slides.add_slide(layout)
            slide.shapes.add_picture(BytesIO(buf), Inches(0), Inches(0), width=prs.slide_width, height=prs.slide_height)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(args.out.resolve()))
        print("Wrote", args.out.resolve(), f"({len(pngs)} slides)")


if __name__ == "__main__":
    main()
