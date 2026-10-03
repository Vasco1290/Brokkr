"""Compare two sets of labels field by field, and allow only the differences a remake may cause.

Usage:  python scripts/48_compare_labels.py OLD NEW      (e.g. published/labels labels)
Needs:  OLD/<model>/label.json and NEW/<model>/label.json for the same models

Used when labels are made again from the same records by newer code (docs/label_schema.md, second note of
4 October 2026). The only differences allowed are:
- a source file's fingerprint: a "sha256" inside a "sources" entry;
- which commit made the label, and when: generated.commit and generated.date_utc.
Everything else (every number, interval, state, ID, build checksum and text) must be equal. Prints, for
each model, how many fields differ in each allowed group and any other difference, then PASS or FAIL.
The rendered label.html and label.md are checked against their label.json by scripts/40_check_labels.py.
"""

import argparse
import json
import sys
from pathlib import Path

ALLOWED = {"source fingerprint", "generated.commit", "generated.date_utc"}


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


def group(path: tuple) -> str:
    if path[-1] == "sha256" and "sources" in path:
        return "source fingerprint"
    if path in (("generated", "commit"), ("generated", "date_utc")):
        return ".".join(path)
    return "other: " + "/".join(str(p) for p in path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("old")
    parser.add_argument("new")
    args = parser.parse_args()
    old_models = sorted(p.parent.name for p in Path(args.old).glob("*/label.json"))
    new_models = sorted(p.parent.name for p in Path(args.new).glob("*/label.json"))
    if old_models != new_models or not old_models:
        sys.exit(f"FAIL: different models: {old_models} vs {new_models}")
    ok = True
    for model in old_models:
        old, new = (json.loads((Path(folder) / model / "label.json").read_text(encoding="utf-8"))
                    for folder in (args.old, args.new))
        counts, others = {}, []
        for path, a, b in differences(old, new):
            g = group(path)
            counts[g] = counts.get(g, 0) + 1
            if g not in ALLOWED:
                others.append(f"{g}: {a!r} -> {b!r}")
        ok &= not others
        commit = f"{old['generated']['commit'][:7]} -> {new['generated']['commit'][:7]}"
        allowed = ", ".join(f"{g} {counts[g]}" for g in sorted(counts) if g in ALLOWED) or "none"
        print(f"{'PASS' if not others else 'FAIL'} {model}: commit {commit}; differing fields: {allowed}"
              + (f"; OTHER differences: {len(others)}" if others else ""))
        for line in others[:10]:
            print(f"     {line}")
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(old_models)} labels in {args.new} equal those in {args.old} "
          f"apart from source fingerprints and the commit that made them")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
