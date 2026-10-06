"""Build the website from the released labels and the figures, check it, and write it.

Usage:  python scripts/49_build_site.py [--open | --serve] [--check]
Needs:  published/labels/*/label.json, docs/figures/ (both committed); nothing in results/ or models/
Writes: site_v0/ (gitignored): every page, the stylesheets, script, fonts, figures and the raw labels

Refuses to build from a dirty working tree (like the measurement scripts): every page's footer names the
commit it was built from, so that commit must hold exactly what built it. Before writing, runs every check
in web/site_checks.py (numbers, meter, themes, links, no JavaScript needed, footer, IDs and the finding's
claim) and prints PASS / FAIL; writes nothing on a FAIL.

--check   build in memory and run the checks only (allowed on a dirty tree, since nothing is written)
--open    afterwards, open the landing page in the default browser (from the file system; no server)
--serve   afterwards, serve site_v0/ at http://127.0.0.1:8000/ until stopped (Ctrl+C)
"""

import argparse
import functools
import http.server
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

import site_build  # noqa: E402  (web/site_build.py)
import site_checks  # noqa: E402

import brokkr_edge  # noqa: E402

OUT = ROOT / "site_v0"


def git(*args) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    dirty = site_build.dirty_tree_problem(git("status", "--porcelain"))
    if dirty and not args.check:
        sys.exit(f"FAIL: {dirty}")
    commit = git("rev-parse", "--short=7", "HEAD")
    labs, figures = site_build.load_inputs()
    pages, files = site_build.build(labs, figures, commit)
    results = site_checks.run_all(
        pages, files, labs, figures, commit, brokkr_edge.__version__, site_build.CLAIMS
    )
    ok = True
    for what, problems in results:
        ok &= not problems
        print(f"{'PASS' if not problems else 'FAIL'}: site, {what}")
        for p in problems[:10]:
            print(f"     {p}")
    print(
        f"{len(pages)} pages, {len(files)} files, built at commit {commit}"
        + (" (with uncommitted changes; check only)" if dirty else "")
    )
    if not ok:
        sys.exit(1)
    if args.check:
        return
    if OUT.exists():
        shutil.rmtree(OUT)
    site_build.write(files, OUT)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}/")
    if args.open:
        webbrowser.open((OUT / "index.html").as_uri())
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
        print("serving http://127.0.0.1:8000/ (Ctrl+C to stop)")
        http.server.ThreadingHTTPServer(("127.0.0.1", 8000), handler).serve_forever()


if __name__ == "__main__":
    main()
