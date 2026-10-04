"""Checks on the built website (docs/website_v0_plan.md, sections 3, 6 and 12). Each returns its problems
as sentences; an empty list is a PASS. scripts/49_build_site.py runs all of them and tests/test_site.py
runs them on a fresh build.

1. numbers: every number a reader sees on a page (text, hover notes, alt text, the texts the condition keys
   show) is a number from the labels or figure entries that page names, in one of their own formats, or
   is on the page's short allow-list (each entry with its reason);
2. meter: every needle angle, interval band, readout text and verdict on the landing page equals the one
   recomputed from a fresh read of the label; the keys are exactly the four chosen conditions;
3. themes: web/themes.py's rules, and the stylesheet uses colour tokens only;
4. links: every internal link, image, stylesheet and script resolves to a built file, every #anchor to an
   id on its page; nothing is loaded from another site;
5. no JavaScript needed: no content is hidden until a script runs, and every number the script can show is
   also on the page without it;
6. every page states the commit it was built from, the brokkr-edge version and the label schema version;
7. build IDs and addresses are unique; the landing finding has its claims-register entry.
"""

import json
import posixpath
import re
from html.parser import HTMLParser
from pathlib import Path

import site_meter as M
import themes
from site_labels import LABELS, address, build_id, builds, measurement
from site_pages import FEATURED, METER_CONDITIONS, count, interval

from brokkr_edge import figures as F
from brokkr_edge.label_render import FORMATS, NUMBER, WORDS_WITH_DIGITS, _leaves

VOID = {"area", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
SHOWN_ATTRS = (
    "title",
    "alt",
    "aria-label",
    "data-th",
    "data-cost-text",
    "data-interval-text",
    "data-verdict",
    "data-label",
)
SCRIPT_ATTRS = ("data-cost-text", "data-interval-text", "data-verdict", "data-label")  # texts site.js shows
EXTRA_WORDS = ("H21",)  # words with digits on the site that are not numbers (H21: a prediction's name)


class PageText(HTMLParser):
    """The text a reader can see, each piece marked as shown only with JavaScript (inside a .js-only
    element, or a text site.js shows) or not; also every element's attributes, for the other checks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []  # (tag, inside js-only, inside script/style)
        self.texts = []  # (text, js only)
        self.elements = []  # (tag, attrs)

    def _state(self):
        return self.stack[-1][1:] if self.stack else (False, False)

    def _attrs(self, tag, attrs, js):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        for name in SHOWN_ATTRS:
            if attrs.get(name):
                self.texts.append((attrs[name], js or name in SCRIPT_ATTRS))
        return attrs

    def handle_starttag(self, tag, attrs):
        js, skip = self._state()
        attrs = self._attrs(tag, attrs, js)
        if tag not in VOID:
            js_only = js or "js-only" in (attrs.get("class") or "").split()
            self.stack.append((tag, js_only, skip or tag in ("script", "style")))

    def handle_startendtag(self, tag, attrs):
        self._attrs(tag, attrs, self._state()[0])

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        js, skip = self._state()
        if not skip and data.strip():
            self.texts.append((data, js))


def parse(page_html: str) -> PageText:
    p = PageText()
    p.feed(page_html)
    return p


def is_page(path: str) -> bool:
    return path.endswith(".html") and not path.endswith("label.html")  # label.html is the label's own render


# ---- 1. numbers ----


def allow_list(page, labs: dict, figures: dict, commit: str, version: str) -> tuple:
    """(strings that are identifiers, not numbers; allowed number tokens) for one page."""
    used = [labs[k] for k in page.labels]
    strings = {s for lab in used for s in _leaves(lab) if isinstance(s, str) and re.search(r"\d", s)}
    strings |= {s[:n] for s in strings if re.fullmatch(r"[0-9a-f]{40,64}", s) for n in (7, 12)}
    strings |= {build_id(lab, builds(lab)[1]) for lab in used} | {address(lab) for lab in used}
    strings |= {commit, f"brokkr-edge {version}"}  # the site's own commit and version (footer)
    allowed = set(page.allowed)
    for lab in used:
        for v in _leaves(lab):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                for kind, fn in FORMATS.items():
                    if kind != "int" or float(v).is_integer():
                        allowed |= set(NUMBER.findall(fn(v)))
                if isinstance(v, int):
                    allowed |= set(NUMBER.findall(count(v)))
    for name in page.figures:
        for e in figures["figures"][name]["values"].values():
            if e["fmt"] == "text":
                strings.add(str(e["value"]))
            elif e["fmt"] in F.FORMATS and isinstance(e["value"], (int, float)):
                for x in [e["value"], *e.get("ci95", [])]:
                    allowed |= set(NUMBER.findall(F.FORMATS[e["fmt"]](x)))
    return sorted(strings, key=len, reverse=True), allowed


def check_numbers(pages: dict, files: dict, labs: dict, figures: dict, commit: str, version: str) -> list:
    problems = []
    for path, page in pages.items():
        strings, allowed = allow_list(page, labs, figures, commit, version)
        loose = set()
        for text, _ in parse(files[path]).texts:
            for s in [*strings, *WORDS_WITH_DIGITS, *EXTRA_WORDS]:
                text = text.replace(s, " ")
            loose |= {t for t in NUMBER.findall(text) if t not in allowed}
        if loose:
            problems.append(f"{path}: numbers not in its labels or figures: {sorted(loose)[:10]}")
    return problems


# ---- 2. the meter ----


def check_meter(files: dict) -> list:
    """Every key's needle, band, texts and verdict, recomputed from the label as it is on disk."""
    label = json.loads((LABELS / FEATURED / "label.json").read_text(encoding="utf-8"))
    lab = builds(label)[1]
    dom = M.domain(label, METER_CONDITIONS)
    p = parse(files["index.html"])
    keys = [a for tag, a in p.elements if tag == "input" and a.get("name") == "cond"]
    problems = []
    if [k["value"] for k in keys] != METER_CONDITIONS:
        problems.append(f"meter keys are {[k['value'] for k in keys]}, not the four chosen conditions")
    names = {c["condition_id"]: c["label"] for c in label["conditions"]}
    for k in keys:
        cid = k["value"]
        cost = measurement(label, "shrinking_cost", lab, cid)
        kind, icon, text = M.verdict(label, cost, cid)
        expected = {
            "data-angle": f"{M.angle(cost['value'], dom):.3f}",
            "data-band": M.band(cost, dom),
            "data-cost-text": FORMATS["pts"](cost["value"]),
            "data-interval-text": interval(cost, "pts"),
            "data-verdict": text,
            "data-verdict-kind": kind,
            "data-label": names[cid],
        }
        for attr, value in expected.items():
            if k.get(attr) != value:
                problems.append(f"meter {cid}: {attr} is {k.get(attr)!r}, the label gives {value!r}")
    clean = measurement(label, "shrinking_cost", lab, "clean")
    needle = next((a for tag, a in p.elements if a.get("id") == "meter-needle"), {})
    if needle.get("data-angle") != f"{M.angle(clean['value'], dom):.3f}":
        problems.append("meter: the needle without JavaScript does not point at the clean shrinking cost")
    if f'id="r-cost">{FORMATS["pts"](clean["value"])}<' not in files["index.html"]:
        problems.append("meter: the readout without JavaScript does not show the clean shrinking cost")
    return problems


# ---- 3. themes ----


def check_themes(files: dict) -> list:
    problems = list(themes.check())
    css = files["assets/site.css"].read_text(encoding="utf-8")
    raw = re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", re.sub(r"/\*.*?\*/", "", css, flags=re.S))
    if raw:
        problems.append(f"assets/site.css uses colours that are not theme tokens: {raw[:5]}")
    return problems


# ---- 4. links ----


def check_links(files: dict) -> list:
    problems = []
    ids = {
        path: {a["id"] for _, a in parse(text).elements if a.get("id")}
        for path, text in files.items()
        if path.endswith(".html") and isinstance(text, str)
    }
    for path, text in files.items():
        if not (path.endswith(".html") and isinstance(text, str)):
            continue
        for tag, a in parse(text).elements:
            targets = [a[k] for k in ("href", "src") if a.get(k)] + [
                s.strip().split(" ")[0] for s in (a.get("srcset") or "").split(",") if s.strip()
            ]
            for t in targets:
                if re.match(r"https?://", t):
                    if tag != "a":
                        problems.append(f"{path}: <{tag}> loads {t} from another site")
                    continue
                target, _, anchor = t.partition("#")
                target = (
                    posixpath.normpath(posixpath.join(posixpath.dirname(path), target)) if target else path
                )
                if target not in files:
                    problems.append(f"{path}: link to {t} goes nowhere")
                elif anchor and anchor not in ids.get(target, set()):
                    problems.append(f"{path}: link to {t}: no id {anchor!r} on {target}")
    css = files["assets/site.css"].read_text(encoding="utf-8")
    for url in re.findall(r"url\([\"']?([^\"')]+)", css):
        if posixpath.normpath(posixpath.join("assets", url)) not in files:
            problems.append(f"assets/site.css: url({url}) goes nowhere")
    return problems


# ---- 5. no JavaScript needed ----


def check_no_js(files: dict) -> list:
    problems = []
    for path, text in files.items():
        if not (is_page(path) and isinstance(text, str)):
            continue
        p = parse(text)
        if '<html lang="en" class="no-js">' not in text:
            problems.append(f"{path}: the page does not start in no-JavaScript mode")
        hidden = [tag for tag, a in p.elements if "hidden" in a]
        handlers = [tag for tag, a in p.elements if any(k.startswith("on") for k in a)]
        if hidden or handlers:
            problems.append(
                f"{path}: content hidden until a script runs, or inline handlers: {hidden + handlers}"
            )
        without = " ".join(t for t, js in p.texts if not js)
        shown_by_js = {n for t, js in p.texts if js for n in NUMBER.findall(t)}
        missing = sorted(n for n in shown_by_js if n not in NUMBER.findall(without))
        if missing:
            problems.append(f"{path}: numbers shown only with JavaScript: {missing[:8]}")
    return problems


# ---- 6 and 7. versions, IDs, the finding's claim ----


def check_footer(pages: dict, files: dict, commit: str, version: str) -> list:
    problems = []
    for path in pages:
        foot = files[path].split('<footer class="foot">')[-1]
        if f"brokkr-edge {version} at commit {commit}" not in foot or "label schema version" not in foot:
            problems.append(
                f"{path}: the footer does not name the commit, brokkr-edge version and schema version"
            )
        if 'name="viewport"' not in files[path]:
            problems.append(f"{path}: no viewport tag")
    return problems


def check_ids(labs: dict) -> list:
    ids = [build_id(lab, builds(lab)[1]) for lab in labs.values()]
    addresses = [address(lab) for lab in labs.values()]
    dup = {x for x in ids + addresses if (ids + addresses).count(x) > 1}
    return [f"build IDs or addresses used twice: {sorted(dup)}"] if dup else []


def check_claim(claims_path: Path) -> list:
    claims = {c["id"]: c for c in json.loads(claims_path.read_text(encoding="utf-8"))["claims"]}
    c = claims.get("site-finding")
    if c is None:
        return ["docs/claims.json has no entry site-finding for the landing page's finding"]
    missing = [
        k for k in ("claim", "where", "scope", "status", "prior_work", "command", "check") if not c.get(k)
    ]
    script = (c.get("command") or "").split()[1:2]
    if missing or not script or not (claims_path.parents[1] / script[0]).exists():
        return [f"claim site-finding is incomplete or names a missing script ({missing})"]
    return []


def run_all(
    pages: dict, files: dict, labs: dict, figures: dict, commit: str, version: str, claims: Path
) -> list:
    """[(what, problems)] for every check."""
    return [
        (
            "numbers come from the labels and figures",
            check_numbers(pages, files, labs, figures, commit, version),
        ),
        ("meter needle, band, readout and verdicts match the label", check_meter(files)),
        ("theme rules (contrast, accent vs INT8 blue, fixed colours, tokens only)", check_themes(files)),
        ("internal links resolve; nothing loaded from other sites", check_links(files)),
        ("pages work with JavaScript off", check_no_js(files)),
        (
            "every page names commit, brokkr-edge version and schema version",
            check_footer(pages, files, commit, version),
        ),
        (
            "build IDs and addresses unique; the finding is in the claims register",
            check_ids(labs) + check_claim(claims),
        ),
    ]

