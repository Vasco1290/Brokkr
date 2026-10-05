"""Wording checks on what Brokkr publishes (labels, the website, the README and the docs).

official_overclaims(): the ImageNet-C conditions were made with the imagecorruptions package v1.1.2
(an extension of the ImageNet-C code) with a one-line fix for NumPy 2, not with "the official" code, and
they are not directly comparable to the released ImageNet-C files (H, 4 October 2026, fix list item 1).
So no sentence may say "official" together with "ImageNet-C" or "corruption code".

A sentence ends at ". ", "! " or "? ", at an HTML tag, at a blank line, and where a Markdown list item,
heading, table row or quote starts; a single line break inside a paragraph does not end it, so a claim
wrapped across two lines is still one sentence. Words inside JSON keys and values ("kind": "official") are
checked one string at a time (see strings_of), so a label's source kind never meets another field's text.

check_docs(): the same check on the docs. A pre-registered file is never edited above its outcomes, so
docs/hypotheses_stage4.md carries a dated correction note (H, 6 October 2026): the check reads only the text
below that note, and reports that the note exists and how many sentences above it it does not read. Three
docs are not read, because they record the fix and quote the old wording (QUOTING_DOCS, each with its reason).
"""

import html
import re
from pathlib import Path

OFFICIAL = re.compile(r"\bofficial\b", re.I)
IMAGENET_C = re.compile(r"imagenet-c|corruption code", re.I)
BREAK = "\x00"
CORRECTION_NOTE = '## Correction note added 6 October 2026: the ImageNet-C code was not "official"'
PUBLIC_DOCS = (
    "README.md",
    "ROADMAP.md",
    "CLAUDE.md",
    "docs/hypotheses.md",
    "docs/hypotheses_stage3.md",
    "docs/hypotheses_stage4.md",
    "docs/stage3_final_run_log.md",
    "docs/stage4_story.md",
    "docs/writeup.md",
    "docs/related_work.md",
)
QUOTING_DOCS = {
    "STATUS.md": "the project log: it records H's fix list, which quotes the old wording",
    "docs/label_schema.md": "its note of 5 October 2026 quotes the old wording it replaced",
    "docs/website_v0_plan.md": "its section 15 records the fix and names the old wording",
}


def sentences(text: str) -> list:
    """The sentences of a text (see the module docstring for where a sentence ends)."""
    text = re.sub(r"<(style|script)\b.*?</\1>", BREAK, text, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", BREAK, text))
    text = re.sub(r"(?m)^([ \t]*#{1,6} [^\n]*)$", lambda m: m.group(1) + BREAK, text)  # a heading ends there
    parts = re.split(rf"(?<=[.!?])\s+|{BREAK}|\n\s*\n|\n(?=[ \t]*(?:[-*+#|>]|\d+\.)\s)", text)
    return [" ".join(p.split()) for p in parts if p and p.strip()]


def official_overclaims(text: str) -> list:
    """Every sentence in `text` that says "official" together with "ImageNet-C" or "corruption code"."""
    return [s for s in sentences(text) if OFFICIAL.search(s) and IMAGENET_C.search(s)]


def strings_of(data) -> list:
    """Every string in a JSON value (keys and values), one by one."""
    if isinstance(data, dict):
        return [s for k, v in data.items() for s in [k, *strings_of(v)]]
    if isinstance(data, list):
        return [s for v in data for s in strings_of(v)]
    return [data] if isinstance(data, str) else []


def json_overclaims(data) -> list:
    return [s for x in strings_of(data) for s in official_overclaims(x)]


def doc_overclaims(text: str) -> tuple:
    """(sentences found, whether the correction note is there, sentences above the note not read). The note's
    own section quotes the old wording to correct it, so the check starts at the next heading after it."""
    if CORRECTION_NOTE in text:
        above, rest = text.split(CORRECTION_NOTE, 1)
        below = rest.split("\n## ", 1)[1] if "\n## " in rest else ""
        return official_overclaims(below), True, len(official_overclaims(above))
    return official_overclaims(text), False, 0


def check_docs(root: Path = Path(".")) -> tuple:
    """(problems, report lines) for PUBLIC_DOCS."""
    problems, report = [], []
    for name in PUBLIC_DOCS:
        found, note, ignored = doc_overclaims((root / name).read_text(encoding="utf-8"))
        problems += [f'{name}: "official" with ImageNet-C or corruption code: {s!r}' for s in found]
        if note:
            report.append(
                f"{name}: correction note of 6 October 2026 found; {ignored} sentence(s) above it "
                "(pre-registered text, not edited) are not read"
            )
    report += [f"{name}: not read ({why})" for name, why in QUOTING_DOCS.items()]
    return problems, report
