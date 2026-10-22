# -*- coding: utf-8 -*-
"""Replace SLIDES in build_deck.py with _deck_slides_body.txt (single source for slide HTML)."""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build_deck.py"
BODY = HERE / "_deck_slides_body.txt"


def main() -> None:
    body = BODY.read_text(encoding="utf-8").strip() + "\n"
    text = BUILD.read_text(encoding="utf-8")
    pattern = re.compile(r'SLIDES = r"""[\s\S]*?"""\n\n\nif __name__ == "__main__":', re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise SystemExit("Could not find SLIDES = r'''...''' block before if __name__")
    def _repl(_m: re.Match[str]) -> str:
        return 'SLIDES = r"""\n' + body + '"""\n\n\nif __name__ == "__main__":'

    text = pattern.sub(_repl, text, count=1)
    text = re.sub(
        r"# 共 \d+ 页",
        "# 共 17 页",
        text,
        count=1,
    )
    BUILD.write_text(text, encoding="utf-8")
    print("Patched", BUILD, "from", BODY)


if __name__ == "__main__":
    main()
