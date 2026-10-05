"""The website generator (Platform plan step 2; docs/website_v0_plan.md, sections 2, 5 and 12).

build() reads only committed files: the released labels (published/labels/), the figures and their
manifest (docs/figures/), the README's checked findings block, ROADMAP.md and docs/related_work.md
(web/site_sources.py), the claims register (docs/claims.json), the themes (web/themes.py), the stylesheet,
script and fonts (web/assets/, web/fonts/) and the favicon (docs/assets/favicon.png). It returns every file
of the site as {path: text or source Path}; no number is computed here.

Each page type (PAGE_TYPES below) makes its pages; this file wraps them in the shared shell:
header with the navigation (built from the page types, so the nav shows only pages that exist), the theme
picker, and a footer naming the commit the site was built from, the brokkr-edge version and the label
schema version. scripts/49_build_site.py runs it, refuses a dirty working tree, checks and writes.
"""

import json
import re
from pathlib import Path

import site_pages as A
import site_pages_more as B
import site_sources
import themes
from site_labels import ROOT, label_dir, label_problems, load_labels
from site_pages import GITHUB, Page, PageType, esc, up

import brokkr_edge
from brokkr_edge.label_schema import SCHEMA_VERSION

FIGURES = ROOT / "docs" / "figures" / "figures.json"
CLAIMS = ROOT / "docs" / "claims.json"
FAVICON = ROOT / "docs" / "assets" / "favicon.png"
ASSETS, FONTS = ROOT / "web" / "assets", ROOT / "web" / "fonts"
THEME_BOOT = (
    "<script>(function(){var d=document.documentElement;d.className='js';var ok=%s,t=null;"
    "try{t=new URLSearchParams(location.search).get('theme')||localStorage.getItem('brokkr-theme')}"
    "catch(e){}if(ok.indexOf(t)>=0)d.setAttribute('data-theme',t)})();</script>"
)


# The page types, in navigation order (section 12e); the nav and the landing page's buttons come from it.
PAGE_TYPES = [
    PageType("landing", None, "index.html", A.landing),
    PageType("catalog", "Catalog", "models/index.html", A.catalog),
    PageType("compare", "Compare", "compare/index.html", B.compare),
    PageType("why-labels", "Why labels?", "why-labels/index.html", B.why_labels),
    PageType("methods", "Methods", "methods/index.html", B.methods),
    PageType("roadmap", "Roadmap", "roadmap/index.html", B.roadmap),
    PageType("model", None, "models/index.html", A.model_pages),  # reached from the catalog, not the nav
]


class Site:
    """What every page type may read: the labels, the figure manifest and the list of page types."""

    def __init__(self, labs: dict, figures: dict, page_types: list, findings: dict, claims: dict):
        self.labs = labs
        self.figures = figures
        self.page_types = {pt.key: pt for pt in page_types}
        self.findings = findings  # the README's checked findings paragraphs (web/site_sources.py)
        self.claims = claims  # docs/claims.json, by id
        self.copies = {}  # site path -> source file, for files a page uses (figures)

    def figure(self, name: str) -> dict:
        return self.figures["figures"][name]

    def svg_size(self, repo_path: str) -> tuple:
        """A committed SVG's width and height, from its own attributes."""
        head = (ROOT / repo_path).read_text(encoding="utf-8")[:400]
        return tuple(int(re.search(rf'{k}="(\d+)"', head).group(1)) for k in ("width", "height"))

    def asset(self, repo_path: str) -> str:
        """Copy a committed file (e.g. 'docs/figures/x.svg') into the site; returns its site path."""
        site_path = "figures/" + Path(repo_path).name
        self.copies[site_path] = ROOT / repo_path
        return site_path


def versions(labs: dict, commit: str) -> str:
    """The footer line: what built the site, and what made its labels (said when they differ)."""
    made = sorted({(lab["generated"]["by"], lab["generated"]["commit"][:7]) for lab in labs.values()})
    schemas = sorted({lab["schema_version"] for lab in labs.values()})
    by_labels = "; ".join(f"{by} at commit {c}" for by, c in made)
    schema = ", ".join(map(str, schemas)) or str(SCHEMA_VERSION)  # a page with no label: the version read
    labels = f" Labels made by {esc(by_labels)}" + (" (different versions)." if len(made) > 1 else ".")
    return (
        f"Built by brokkr-edge {brokkr_edge.__version__} at commit {commit} from label schema version "
        f"{schema}." + (labels if made else "")
    )


def shell(path: str, page: Page, page_types: list, labs: dict, commit: str) -> str:
    """The shared layout around a page's body."""
    u = up(path)
    nav = "".join(
        f'<a href="{u}{pt.path}"{" aria-current=page" if pt.path == path else ""}>{esc(pt.nav)}</a>'
        for pt in page_types
        if pt.nav
    )
    options = "".join(f'<option value="{k}">{esc(t["name"])}</option>' for k, t in themes.THEMES.items())
    used = {k: labs[k] for k in page.labels}
    return f"""<!doctype html>
<html lang="en" class="no-js">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(page.title)}</title>
{THEME_BOOT % json.dumps(list(themes.THEMES))}
<link rel="icon" href="{u}favicon.png">
<link rel="stylesheet" href="{u}assets/themes.css">
<link rel="stylesheet" href="{u}assets/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="nav">
  <a class="brand" href="{u}index.html"><span class="brand-mark" aria-hidden="true"></span>BROKKR</a>
  <nav aria-label="Main">{nav}<a href="{GITHUB}">GitHub</a></nav>
  <label class="theme-pick js-only"><span class="theme-word">Theme</span>
    <select id="theme-picker"><option value="auto">System</option>{options}</select></label>
</header>
<main id="main">
{page.body}
</main>
<footer class="foot">
<p class="mono">{versions(used, commit)}</p>
<p>Brokkr's code: Apache-2.0. Label data: CC BY 4.0. Model weights keep their own licences
(see each label).</p>
</footer>
<script src="{u}assets/site.js"></script>
</body>
</html>
"""


def dirty_tree_problem(git_status: str) -> str | None:
    """Why the site must not be built from this working tree (`git status --porcelain` output), or None.
    The footer names the commit the site was built from, so that commit must hold exactly what built it."""
    if git_status.strip():
        return "the working tree has uncommitted changes; commit them first:\n" + git_status
    return None


def load_inputs() -> tuple:
    """The released labels and the figure manifest; stops on a label the site cannot read."""
    labs = load_labels()
    problems = label_problems(labs)
    if problems:
        raise SystemExit("FAIL: labels the site cannot use:\n  " + "\n  ".join(problems))
    return labs, json.loads(FIGURES.read_text(encoding="utf-8"))


def build(labs: dict, figures: dict, commit: str, page_types: list = None) -> tuple:
    """Every page and file of the site. Returns (pages {path: Page}, files {path: str | Path})."""
    page_types = page_types if page_types is not None else PAGE_TYPES
    claims = {c["id"]: c for c in json.loads(CLAIMS.read_text(encoding="utf-8"))["claims"]}
    site = Site(labs, figures, page_types, site_sources.findings(), claims)
    pages = {}
    for pt in page_types:
        for path, page in pt.make(site):
            if path in pages:
                raise SystemExit(f"FAIL: two pages want the same address: {path}")
            pages[path] = page
    files = {path: shell(path, page, page_types, labs, commit) for path, page in pages.items()}
    files["assets/themes.css"] = themes.css()
    files["assets/site.css"] = ASSETS / "site.css"
    files["assets/site.js"] = ASSETS / "site.js"
    files["favicon.png"] = FAVICON
    files[".nojekyll"] = ""  # GitHub Pages: serve the files as they are
    for font in sorted(FONTS.rglob("*")):
        if font.is_file():
            files["fonts/" + font.relative_to(FONTS).as_posix()] = font
    for name, lab in labs.items():  # the raw label beside its page (CC BY 4.0, committed)
        for file in ("label.json", "label.html"):
            files[f"{label_dir(lab)}/{file}"] = ROOT / "published" / "labels" / name / file
    files.update(site.copies)
    return pages, files


def write(files: dict, out: Path) -> None:
    for path, content in files.items():
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            target.write_bytes(content.read_bytes())
        else:
            target.write_text(content, encoding="utf-8", newline="\n")
