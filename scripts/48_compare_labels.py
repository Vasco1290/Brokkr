"""Compare two sets of labels field by field, and allow only the differences a remake may cause or that a
dated entry in docs/label_changes.json expects.

Usage:  python scripts/48_compare_labels.py OLD NEW      (e.g. published/labels labels)
Needs:  OLD/<model>/label.json and NEW/<model>/label.json for the same models (label.html beside them is
        compared too, when both sets have it)

Used when labels are made again from the same records by newer code (docs/label_schema.md, second note of
4 October 2026). Differences always allowed:
- a source file's fingerprint: a "sha256" beside a "file" key (every source reference, in "sources"
  lists and in hardware[].fingerprint_source); a model file's checksum (builds[].file.sha256) has no
  "file" key beside it, so it must not change;
- which commit made the label, and when: generated.commit and generated.date_utc.
Differences allowed only when listed (docs/label_changes.json, H's request of 5 October 2026): each entry is
dated, points to the note that decided it, and names one label.json field, or one piece of text the label
renderer writes into label.html, with its exact old and new text. A listed label.json change passes only
where the old label holds exactly the old text and the new label exactly the new text.
label.html: the old label rendered by today's renderer must equal the old label.html with only the listed
renderer changes applied (so the renderer changed in no other way), and the new label.html must equal the
new label rendered by today's renderer.
Everything else (every number, interval, state, ID, build checksum and text) must be equal. Prints, for each
model, the allowed differences and each accepted listed change by name, any other difference, then PASS or
FAIL.
"""

import argparse
import html
import json
import sys
from pathlib import Path

from brokkr_edge.label_render import to_html

ALLOWED = {"source fingerprint", "generated.commit", "generated.date_utc"}
CHANGES = Path("docs/label_changes.json")


def differences(old, new, path=()):
    """Every (path, old value, new value) where the two JSON values differ."""
    if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
        for key in old:
            yield from differences(old[key], new[key], (*path, key))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            yield from differences(a, b, (*path, i))
    elif old != new:
        yield path, old, new


def parent(label: dict, path: tuple):
    for part in path[:-1]:
        label = label[part]
    return label


def group(label: dict, path: tuple) -> str:
    if path[-1] == "sha256" and "file" in parent(label, path):
        return "source fingerprint"
    if path in (("generated", "commit"), ("generated", "date_utc")):
        return ".".join(path)
    return "other: " + "/".join(str(p) for p in path)


def name_of(change: dict) -> str:
    where = (
        "label.json " + "/".join(map(str, change["path"])) if change["in"] == "label.json" else change["in"]
    )
    return f"[{change['date']}] {change['what']} ({where}; {change['note']})"


def listed(changes: list, path: tuple, a, b) -> dict | None:
    """The listed label.json change that this difference is, or None."""
    for c in changes:
        if c["in"] == "label.json" and tuple(c["path"]) == path and (a, b) == (c["old"], c["new"]):
            return c
    return None


def compare_html(old: dict, new: dict, old_html: str, new_html: str, changes: list) -> tuple:
    """(accepted renderer changes, problems) for one model's label.html."""
    accepted, expected_old = [], old_html
    for c in changes:
        if c["in"] != "label.html":
            continue
        before = expected_old
        for a, b in {(c["old"], c["new"]), (html.escape(c["old"]), html.escape(c["new"]))}:
            expected_old = expected_old.replace(a, b)
        if expected_old != before:
            accepted.append(c)
    problems = []
    if to_html(old) != expected_old:
        problems.append("label.html: the renderer changed in a way docs/label_changes.json does not list")
    if to_html(new) != new_html:
        problems.append("label.html: the new label.html is not today's render of the new label.json")
    return accepted, problems


def compare_model(old_dir: Path, new_dir: Path, model: str, changes: list) -> tuple:
    """(ok, summary line, detail lines) for one model."""
    old, new = (
        json.loads((d / model / "label.json").read_text(encoding="utf-8")) for d in (old_dir, new_dir)
    )
    counts, accepted, others = {}, [], []
    for path, a, b in differences(old, new):
        g = group(old, path)
        c = listed(changes, path, a, b)
        if c is not None:
            accepted.append(c)
        elif g in ALLOWED:
            counts[g] = counts.get(g, 0) + 1
        else:
            others.append(f"{g}: {a!r} -> {b!r}")
    pages = [d / model / "label.html" for d in (old_dir, new_dir)]
    if all(p.exists() for p in pages):
        more, problems = compare_html(old, new, *(p.read_text(encoding="utf-8") for p in pages), changes)
        accepted += more
        others += problems
    commit = f"{old['generated']['commit'][:7]} -> {new['generated']['commit'][:7]}"
    allowed = ", ".join(f"{g} {counts[g]}" for g in sorted(counts)) or "none"
    line = (
        f"{'PASS' if not others else 'FAIL'} {model}: commit {commit}; differing fields: {allowed}; "
        f"accepted listed changes: {len(accepted)}"
        + (f"; OTHER differences: {len(others)}" if others else "")
    )
    details = [f"     accepted: {name_of(c)}" for c in accepted] + [f"     {x}" for x in others[:10]]
    return not others, line, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    parser.add_argument("--changes", type=Path, default=CHANGES)
    args = parser.parse_args()
    changes = json.loads(args.changes.read_text(encoding="utf-8"))["changes"]
    old_dir, new_dir = Path(args.old), Path(args.new)
    old_models = sorted(p.parent.name for p in old_dir.glob("*/label.json"))
    new_models = sorted(p.parent.name for p in new_dir.glob("*/label.json"))
    if old_models != new_models or not old_models:
        sys.exit(f"FAIL: different models: {old_models} vs {new_models}")
    ok = True
    for model in old_models:
        model_ok, line, details = compare_model(old_dir, new_dir, model, changes)
        ok &= model_ok
        print(line)
        for d in details:
            print(d)
    print(
        f"\n{'PASS' if ok else 'FAIL'}: {len(old_models)} labels in {args.new} equal those in {args.old} "
        "apart from source fingerprints, the commit that made them and the changes listed in "
        f"{args.changes.as_posix()}"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
