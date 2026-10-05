"""The website's committed sources other than the labels and figures (docs/website_v0_plan.md, section 14).

- findings(): the Stage 4 findings, read word for word from the generated block between the "findings"
  markers in README.md. scripts/45_readme_findings.py writes that block from the 4.1 records and the
  labels, and its --check (run by scripts/40_check_labels.py) fails if the block differs from a fresh
  render, so the site quotes checked text without needing the records (which are not committed).
- roadmap(): ROADMAP.md's public stages and Platform plan steps with their status marks, and its public
  "Later versions" items, under plain names without stage or step numbers or dates (H, 3 and 4 October
  2026). Each item is marked public or internal in ROADMAP.md; an unmarked item stops the build.
- related_work(): the related work shown on the site, read from docs/related_work.md (verified on each
  source's own page on 3 October 2026; tracked since 6 October 2026): title, authors and year from each
  entry's first line, key, venue and link from its "Site:" line. Bibliographic facts only, no claims.
"""

import re

from site_labels import ROOT

README = ROOT / "README.md"
ROADMAP = ROOT / "ROADMAP.md"
FINDINGS_START, FINDINGS_END = "<!-- findings:start", "<!-- findings:end -->"
# Each findings paragraph, known by its first words (scripts/45_readme_findings.py writes them in this order).
FINDINGS = {
    "overview": " torchvision models,",
    "headline": "Consistent with prior work on quantized models",
    "prereg": "Pre-registered (",
    "example": "For example,",
    "explore": "Exploratory, not pre-registered: across all",
    "noise_blur": "Exploratory, not pre-registered: large shrinking costs",
    "why": "The tests of *why*",
}


def findings_block(text: str | None = None) -> str:
    text = README.read_text(encoding="utf-8") if text is None else text
    return text.split(FINDINGS_START)[1].split("-->", 1)[1].split(FINDINGS_END)[0]


def findings(text: str | None = None) -> dict:
    """{key: paragraph} for every findings bullet, unwrapped to one line each."""
    bullets = [" ".join(b.split()) for b in re.split(r"\n- ", "\n" + findings_block(text).strip())[1:]]
    out = {}
    for key, start in FINDINGS.items():
        found = [b for b in bullets if (start in b[:40] if key == "overview" else b.startswith(start))]
        if len(found) != 1:
            raise SystemExit(
                f"FAIL: README findings: expected one paragraph starting {start!r}, found {len(found)}"
            )
        out[key] = found[0]
    return out


# ---- the roadmap ----

STATUS = {"[x]": "done", "[~]": "in progress", "[ ]": "not started"}
HEADING = re.compile(r"^(#{2,3}) (.+?) `(\[[x~ ]\])`")
# Every stage, Platform plan step and "Later versions" item ends with one of these (H, 4 October 2026).
MARK = re.compile(r"<!-- (public|internal)(?:: (.+?))? -->\s*$")
# What a public name may not hold: a stage, step or task number, or a plan code ("Raspberry Pi 5" is a name).
NUMBERED = re.compile(r"\b(?:stage|step|task)\s*\d|\b\d+\.\d+\b|\bP\d\b|^\d+b?\.", re.I)


def clean_title(title: str) -> str:
    """'Stage 1 — Core measurement' -> 'Core measurement'; '1. Labels (P1)' -> 'Labels'; backticks dropped."""
    title = re.sub(r"\s*\(P\d\)", "", title).replace("`", "")
    title = re.sub(r"^Stage \d+\s*[—-]\s*", "", title)
    return re.sub(r"^\d+b?\.\s*", "", title)


def undated(text: str) -> str:
    """A 'Later versions' item without its bracketed notes about dates, people, STATUS.md or plan codes."""
    text = re.sub(r"\s*\([^()]*(?:20\d\d|STATUS|by H|P\d)[^()]*\)", "", text).replace("`", "")
    return text.strip()


def public_name(line: str, default: str) -> str | None:
    """The item's name on the website, or None if it is internal; stops on an item with no mark, or a
    public name with a stage, step or task number in it."""
    m = MARK.search(line)
    if m is None:
        raise SystemExit(f"FAIL: ROADMAP.md item has no <!-- public --> or <!-- internal --> mark: {line!r}")
    if m.group(1) == "internal":
        return None
    name = m.group(2) or default
    if NUMBERED.search(name):
        raise SystemExit(f"FAIL: ROADMAP.md public name has a stage, step or task number: {name!r}")
    return name


def roadmap(text: str | None = None) -> dict:
    """{'built', 'now', 'next', 'later', 'later_versions'}: the public items' plain names, in file order."""
    text = ROADMAP.read_text(encoding="utf-8") if text is None else text
    lines = text.splitlines()
    stages, steps, later_versions = [], [], []
    section = None
    for i, line in enumerate(lines):
        if line.startswith("## "):
            section = (
                "platform" if line.startswith("## Platform plan") else "stages" if "Stage" in line else None
            )
        m = HEADING.match(line)
        if m and section == "stages" and m.group(1) == "##":
            stages.append((public_name(line, clean_title(m.group(2))), STATUS[m.group(3)]))
        elif m and section == "platform" and m.group(1) == "###":
            steps.append((public_name(line, clean_title(m.group(2))), STATUS[m.group(3)]))
        elif section == "platform" and line.startswith("### Later versions"):
            for item in lines[i + 1 :]:
                if item.startswith("#"):
                    break
                if item.startswith("- "):
                    name = public_name(item, undated(MARK.sub("", item[2:])).rstrip("."))
                    if name:
                        later_versions.append(name)
    stages = [s for s in stages if s[0]]
    steps = [s for s in steps if s[0]]
    not_started = [s for s in steps if s[1] == "not started"]
    return {
        "built": [s[0] for s in stages + steps if s[1] == "done"],
        "now": [s[0] for s in stages + steps if s[1] == "in progress"],
        "next": [s[0] for s in not_started[:1]],
        "later": [s[0] for s in not_started[1:]],
        "later_versions": later_versions,
    }


# ---- related work ----

RELATED_WORK = ROOT / "docs" / "related_work.md"
ENTRY = re.compile(r"^\d+\. \*\*(?P<title>.+?)\.\*\* (?P<authors>.+?)\. (?P<year>\d{4}) \(")
SITE_LINE = re.compile(
    r"Site: key (?P<key>\S+) · (?P<shown>cited|also read) · where (?P<where>.+?) · link (?P<link>\S+)$"
)


def related_work(text: str | None = None) -> dict:
    """{"cited": [...], "also read": [...]}: (key, authors, year, title, where, link) for every entry of
    docs/related_work.md that the site shows, in the file's order. Title, authors and year come from the
    entry's first line, the rest from its "Site:" line; an entry without a readable "Site:" line stops the
    build ("Site: not shown" leaves it off the site)."""
    text = RELATED_WORK.read_text(encoding="utf-8") if text is None else text
    out = {"cited": [], "also read": []}
    for block in re.split(r"\n(?=\d+\. \*\*)", text)[1:]:
        lines = [x.strip() for x in block.split("\n\n")[0].splitlines()]
        site = next((x for x in lines if x.startswith("Site:")), None)
        if site is None:
            raise SystemExit(f"FAIL: docs/related_work.md entry has no Site: line: {lines[0]!r}")
        if site.startswith("Site: not shown"):
            continue
        entry, s = ENTRY.match(" ".join(x for x in lines if not x.startswith("Site:"))), SITE_LINE.match(site)
        if entry is None or s is None:
            raise SystemExit(f"FAIL: docs/related_work.md entry cannot be read: {lines[0]!r}")
        out[s["shown"]].append(
            (s["key"], entry["authors"], entry["year"], entry["title"], s["where"], s["link"])
        )
    return out
