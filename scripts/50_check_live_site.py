"""Check the published website against the local build (docs/publishing_plan.md, step 5).

Usage:  python scripts/50_check_live_site.py
Needs:  site_v0/ built by scripts/49_build_site.py from the commit that was published, and the network

Fetches the landing page, one model page, the compare page and a missing address (which must answer 404
with the site's 404.html), then every file of the local build, and prints PASS only if each answers and
equals the local file byte for byte.
"""

import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://vasco1290.github.io/Brokkr/"
SITE = Path("site_v0")
BASE_TAG = b'<base href="/Brokkr/">'


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Brokkr live check)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")


def title(body):
    m = re.search(rb"<title>(.*?)</title>", body)
    return m.group(1).decode() if m else None


ok = True
print("1. The pages asked for")
pages = [
    ("landing", "", "index.html"),
    (
        "model page (MobileNetV3-Large)",
        "models/mobilenet-v3-large/int8-percentile-99.99/",
        "models/mobilenet-v3-large/int8-percentile-99.99/index.html",
    ),
    ("compare", "compare/", "compare/index.html"),
]
for name, path, local in pages:
    status, body, ctype = get(BASE + path)
    same = body == (SITE / local).read_bytes()
    ok &= status == 200 and same
    print(
        f"   {name}: {BASE + path} -> HTTP {status}, {ctype}, title {title(body)!r}, {len(body):,} bytes, "
        f"equal to the local build: {same}"
    )
missing = BASE + "models/no-such-model/deep/"
status, body, ctype = get(missing)
same = body == (SITE / "404.html").read_bytes()
says = b"This page doesn't exist." in body
ok &= status == 404 and same
print(
    f"   missing address: {missing} -> HTTP {status}, title {title(body)!r}, equal to the local 404.html: "
    f"{same}, says the page doesn't exist: {says}, carries the /Brokkr/ base: {BASE_TAG in body}"
)

print("2. Every built file, byte for byte")
files = sorted(p.relative_to(SITE).as_posix() for p in SITE.rglob("*") if p.is_file())
bad = []
for f in files:
    status, body, _ = get(BASE + f)
    if status != 200 or body != (SITE / f).read_bytes():
        bad.append((f, status))
ok &= not bad
print(
    f"   {len(files) - len(bad)} of {len(files)} files answered 200 and equal the local build"
    + (f"; not equal: {bad}" if bad else "")
)
print(f"PASS: the live site at {BASE} equals site_v0/" if ok else "FAIL")
sys.exit(0 if ok else 1)
