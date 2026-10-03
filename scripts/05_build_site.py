"""Build the results web page from everything in results/ and models/.

Usage:  python scripts/05_build_site.py
Writes: site/index.html (gitignored; published separately to the gh-pages branch)
"""

import sys

from brokkr_edge.report import build_site

out = build_site("results", "models", "site")
page = out.read_text(encoding="utf-8")
print(f"Wrote {out} ({len(page) / 1000:.0f} KB)")

# Sanity checks: the page has content and no leftover template placeholders.
passed = "<table>" in page and "{" not in page.split("<body>")[1]
print("PASS" if passed else "FAIL: page has no tables or contains unfilled placeholders")
sys.exit(0 if passed else 1)
