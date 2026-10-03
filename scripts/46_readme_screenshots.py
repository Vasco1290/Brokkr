"""Make the README's label screenshots (light and dark) from the released label.

Usage:  python scripts/46_readme_screenshots.py [--browser PATH]
Needs:  published/labels/mobilenet_v3_large/label.html (committed), and Chrome or Edge on this machine
Writes: docs/assets/label-summary-light.png, docs/assets/label-summary-dark.png, and
        docs/assets/label-summary.json (which label file the pictures show, with its SHA-256)

The page is the committed label.html with one added style rule that hides everything after the summary
(the label's own content is not changed), shown by a headless browser at a fixed width with the
light or dark colour scheme, then trimmed of empty background at the bottom. The pictures show numbers,
so tests/test_readme.py fails if the label they were made from has changed since (the SHA-256 of its
text, with line endings as LF, in the JSON). The pictures are for viewing only; the README's numbers
come from its generated blocks.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

LABEL = Path("published/labels/mobilenet_v3_large/label.html")
OUT = Path("docs/assets")
WIDTH, HEIGHT = 900, 1400  # px; GitHub shows README images at most about this wide
BROWSERS = [
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
]
# Hide every section after the summary: the second <h2> ("What was tested") and everything after it.
SUMMARY_ONLY = "<style>h2~h2,h2~h2~*{display:none!important}</style></head>"


def text_sha256(path: Path) -> str:
    """SHA-256 of the file's text with line endings as LF, so a Windows and a Linux checkout agree."""
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def trim_bottom(path: Path, margin: int = 24) -> Image.Image:
    """Cut the empty background below the last row that differs from the page's background colour."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    background = a[-1, -1]
    used = np.nonzero(np.abs(a - background).sum(axis=2).max(axis=1) > 30)[0]
    return im.crop((0, 0, im.width, min(im.height, int(used.max()) + margin)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", type=Path)
    args = parser.parse_args()
    browser = args.browser or next((b for b in BROWSERS if b.exists()), None)
    if browser is None:
        sys.exit("FAIL: no Chrome or Edge found; pass --browser")
    page = LABEL.read_text(encoding="utf-8")
    if page.count("</head>") != 1:
        sys.exit(f"FAIL: {LABEL} does not look like a label page")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "summary.html").write_text(page.replace("</head>", SUMMARY_ONLY), encoding="utf-8")
        for scheme, value in (("light", 1), ("dark", 0)):  # Blink: preferredColorScheme 0 = dark, 1 = light
            shot = tmp / f"{scheme}.png"
            subprocess.run(
                [
                    str(browser),
                    "--headless=new",
                    "--disable-gpu",
                    "--no-first-run",
                    "--hide-scrollbars",
                    f"--user-data-dir={tmp / 'profile'}",
                    f"--window-size={WIDTH},{HEIGHT}",
                    f"--blink-settings=preferredColorScheme={value}",
                    f"--screenshot={shot}",
                    (tmp / "summary.html").as_uri(),
                ],
                check=True,
                timeout=120,
                capture_output=True,
            )
            out = OUT / f"label-summary-{scheme}.png"
            trim_bottom(shot).quantize(colors=128).save(out, optimize=True)
            im = Image.open(out)
            print(f"wrote {out}: {im.width}x{im.height}, {out.stat().st_size} bytes")
    record = {
        "label": LABEL.as_posix(),
        "sha256": text_sha256(LABEL),
        "pictures": [f"label-summary-{s}.png" for s in ("light", "dark")],
        "browser": browser.name,
        "width_px": WIDTH,
        "made_by": "scripts/46_readme_screenshots.py",
    }
    (OUT / "label-summary.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"PASS: screenshots of {LABEL} (SHA-256 {record['sha256'][:12]}) in {OUT}")


if __name__ == "__main__":
    main()
