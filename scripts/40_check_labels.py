"""Check every label against its sources: PASS only if every number, state and render holds up.

Usage:  python scripts/40_check_labels.py [--labels labels]
Needs:  <labels>/<model>/label.json, label.md, label.html (scripts/39_make_labels.py) and every file each
        label names in its "sources"

For each label, independently of the code that made it:
1. It follows docs/label_schema.md version 1 (brokkr_edge.label_schema.check_label).
2. Every source file still has the SHA-256 the label records.
3. Each build's file checksum and size equal a build record it names, and its status equals that
   record's own check (brokkr_edge.schema.check_build_record).
4. Every copied number (value and interval) equals the field it names in its source record.
5. Every damage drop and shrinking cost is recomputed from the saved scores (paired bootstrap, 1,000
   resamples, seed 0) and must be identical.
6. Every envelope state and shrinking-cost flag, and the summary, are recomputed from the measurements
   with the committed rule (brokkr_edge.label) and must be identical; so must the FP32 sanity check.
7. label.md and label.html equal a fresh render of label.json, every number a reader sees in them is a
   label.json number, and the Markdown's model-card metadata parses (huggingface_hub.ModelCard).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import ModelCard

from brokkr_edge.judge import top1_correct
from brokkr_edge.label import envelope_state, paired, shrinking_cost_flag, summary
from brokkr_edge.label_render import to_html, to_markdown, unexplained_numbers
from brokkr_edge.label_schema import check_label
from brokkr_edge.model_list import load_model_list
from brokkr_edge.results import sha256_of
from brokkr_edge.schema import check_build_record

parser = argparse.ArgumentParser()
parser.add_argument("--labels", default="labels")
args = parser.parse_args()


def field(record: dict, path: str):
    for part in path.split("."):
        record = record[part]
    return record


def correct_from(npz_source: dict) -> np.ndarray:
    with np.load(npz_source["file"]) as data:
        return top1_correct(data["logits"], data["labels"])


def check(folder: Path) -> list:
    label = json.loads((folder / "label.json").read_text(encoding="utf-8"))
    problems = list(check_label(label))

    # 2. Source files unchanged (nothing below can be checked without them).
    changed = [s["file"] for s in label["sources"]
               if not Path(s["file"]).exists() or sha256_of(s["file"]) != s["sha256"]]
    if changed:
        return problems + [f"source changed or missing: {f}" for f in changed]

    # 3. Builds match their build records.
    records = [
        json.loads(Path(s["file"]).read_text(encoding="utf-8"))
        for s in label["sources"]
        if s["file"].startswith("models/") and s["file"].endswith(".json")
    ]
    for b in label["builds"]:
        match = [r for r in records if r["file"]["sha256"] == b["file"]["sha256"]]
        if not match or match[0]["file"]["size_bytes"] != b["file"]["size_bytes"]:
            problems.append(f"build {b['build_id']}: no named build record has this file")
        elif check_build_record(match[0], set())[0] != b["status"]:
            problems.append(f"build {b['build_id']}: status differs from its build record's check")

    # 4 and 5. Every measurement.
    for m in label["measurements"]:
        where = f"{m['metric']} {m['build_id']} {m['condition_id']}"
        if m["metric"] in ("damage_drop", "shrinking_cost"):
            new, old = (correct_from(s) for s in m["sources"])
            p = paired(new, old)
            if (p["value"], p["ci95"]) != (m["value"], m["ci95"]):
                problems.append(f"{where}: recomputed {p['value']} {p['ci95']} differs")
            continue
        s = m["sources"][0]
        saved = field(json.loads(Path(s["file"]).read_text(encoding="utf-8")), s["field"])
        if (saved["value"], saved["ci95"]) != (m["value"], m["ci95"]):
            problems.append(f"{where}: differs from {s['file']} {s['field']}")

    # 6. Envelope, flags, summary and the FP32 sanity check, recomputed.
    ix = {(m["metric"], m["build_id"], m["condition_id"]): m for m in label["measurements"]}
    builds = {b["role"]: b for b in label["builds"]}
    rule = label["envelope"]["rule"]
    for row in label["envelope"]["rows"]:
        bid, cid = row["build_id"], row["condition_id"]
        build = next(b for b in label["builds"] if b["build_id"] == bid)
        if build["status"] == "failed":
            expected = ("INT8 build failed", None)
        else:
            state, _ = envelope_state(
                ix[("coverage", bid, cid)]["ci95"], ix[("damage_drop", bid, cid)]["ci95"], rule
            )
            flag = None
            if build["role"] == "labelled":
                fp32 = ix[("top1", builds["reference"]["build_id"], cid)]["value"]
                flag = shrinking_cost_flag(ix[("shrinking_cost", bid, cid)]["ci95"], fp32, rule)
            expected = (state, flag)
        if (row["state"], row["shrinking_cost_flag"]) != expected:
            problems.append(
                f"envelope {bid} {cid}: {row['state']}/{row['shrinking_cost_flag']}, rule gives {expected}"
            )
    labels = {c["condition_id"]: c["label"] for c in label["conditions"]}
    if summary(label["envelope"]["rows"], label["label_id"], labels) != label["summary"]:
        problems.append("summary differs from the one the envelope gives")
    sanity = label["checks"]["fp32_sanity"]
    measured = ix[("top1", builds["reference"]["build_id"], "clean")]["value"]
    published = load_model_list()[label["model"]["name"]]["published"]["top1"]
    if (sanity["measured"], sanity["published"]) != (measured, published) or sanity["pass"] != (
        abs(measured - published) <= sanity["tolerance"]
    ):
        problems.append("FP32 sanity check differs from its sources")

    # 7. Renders.
    md, page = (
        (folder / "label.md").read_text(encoding="utf-8"),
        (folder / "label.html").read_text(encoding="utf-8"),
    )
    if md != to_markdown(label) or page != to_html(label):
        problems.append("label.md or label.html differs from a fresh render of label.json")
    for name, text in (("label.md", md), ("label.html", page)):
        loose = unexplained_numbers(label, text)
        if loose:
            problems.append(f"{name}: numbers not in label.json: {loose[:5]}")
    card = ModelCard(md)
    if card.data.to_dict().get("license") != "other":
        problems.append("the model card's metadata does not parse as expected")
    return problems


folders = sorted(p for p in Path(args.labels).iterdir() if (p / "label.json").exists())
if not folders:
    sys.exit(f"FAIL: no labels in {args.labels}")
all_ok = True
for folder in folders:
    problems = check(folder)
    all_ok &= not problems
    label = json.loads((folder / "label.json").read_text(encoding="utf-8"))
    print(
        f"{'PASS' if not problems else 'FAIL'} {folder.name}: {len(label['measurements'])} measurements, "
        f"{len(label['envelope']['rows'])} envelope rows, {len(label['sources'])} source files checked"
    )
    for p in problems[:20]:
        print(f"     {p}")
print(f"\n{'PASS' if all_ok else 'FAIL'}: {len(folders)} labels in {args.labels}")
sys.exit(0 if all_ok else 1)
