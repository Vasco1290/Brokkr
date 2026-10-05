"""Wording checks on what Brokkr publishes (labels, the website, the README).

official_overclaims(): the ImageNet-C conditions were made with the imagecorruptions package v1.1.2
(an extension of the ImageNet-C code) with a one-line fix for NumPy 2, not with "the official" code, and
they are not directly comparable to the released ImageNet-C files (H, 4 October 2026, fix list item 1).
So no sentence may say "official" together with "ImageNet-C" or "corruption code".

A sentence ends at ". ", "! ", "? ", a line break or an HTML tag, so a heading and the text under it, or
two list items, are never read as one sentence. Words inside JSON keys and values ("kind": "official") are
checked one string at a time (see strings_of), so a label's source kind never meets another field's text.
"""

import html
import re

OFFICIAL = re.compile(r"\bofficial\b", re.I)
IMAGENET_C = re.compile(r"imagenet-c|corruption code", re.I)


def sentences(text: str) -> list:
    """The sentences of a text; HTML tags (and <style>/<script> contents) end a sentence."""
    text = re.sub(r"<(style|script)\b.*?</\1>", "\n", text, flags=re.S | re.I)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", text))
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [" ".join(p.split()) for p in parts if p.strip()]


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
