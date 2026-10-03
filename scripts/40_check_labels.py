"""Check every label against its sources: PASS only if every number, state and render holds up.

Usage:  python scripts/40_check_labels.py [--labels labels]
Needs:  <labels>/<model>/label.json, label.md, label.html (scripts/39_make_labels.py) and every file each
        label names in its "sources"

For each label, independently of the code that made it:
1. It follows docs/label_schema.md version 1 (brokkr_edge.label_schema.check_label).
2. Every source file still has the SHA-256 the label records.
3. Each build's file checksum and size equal a build record it names, and its status equals that
   record's own check (brokkr_edge.schema.check_build_record); a failed build's failure (value, limit,
   images, split) equals that record's sanity check. Display names equal brokkr_edge/model_list.json's.
4. Every copied number (value and interval) equals the field it names in its source record.
5. Every damage drop and shrinking cost is recomputed from the saved scores (paired bootstrap, 1,000
   resamples, seed 0) and must be identical.
6. Every envelope state, failed line and shrinking-cost flag, and the summary, are recomputed from the
   measurements with the committed rule (brokkr_edge.label) and must be identical; so must the FP32
   sanity check.
7. Every measured speed row equals its latency record (p50/p95/p99, spread, the unstable flag and its
   line, sessions and runs, discarded sessions, pinning, VNNI, the timed file), its timed window is
   recomputed from the record's .npz, and INT8's time as a multiple of FP32's is recomputed; every
   runtime equals the runtime block of its source record (docs/label_schema.md, note of 3 October 2026).
8. The licence section is complete and equals the build records (model code and weights licence, the
   training data read from the weights name), with Brokkr's code Apache-2.0 and the label data CC BY 4.0.
9. label.md and label.html equal a fresh render of label.json, every number a reader sees in them is a
   label.json number, and the Markdown's model-card metadata parses (huggingface_hub.ModelCard) with
   license "other" and license_link "#licences".
Then, once: the README's label example equals a fresh render of its two labels
(scripts/41_readme_label_example.py --check).
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import ModelCard

from brokkr_edge.judge import top1_correct
from brokkr_edge.label import envelope_state, failed_lines, paired, shrinking_cost_flags, summary
from brokkr_edge.label_render import to_html, to_markdown, unexplained_numbers
from brokkr_edge.label_schema import check_label
from brokkr_edge.model_list import load_model_list, load_precision_display_names
from brokkr_edge.results import sha256_of
from brokkr_edge.schema import MIN_AGREEMENT_WITH_FP32, check_build_record

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


def utc(seconds: float) -> str:
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).isoformat(timespec="seconds")


def check_speed(label: dict, builds_by_id: dict) -> list:
    """Step 7: speed rows and runtimes against their latency and accuracy records."""
    problems = []
    runtimes = {r["runtime_id"]: r for r in label["runtimes"]}
    for rid, r in runtimes.items():
        record = json.loads(Path(r["sources"][0]["file"]).read_text(encoding="utf-8"))["runtime"]
        same = all(r[k] == record.get(k) for k in ("name", "version", "execution_provider", "threads"))
        same &= all(r[k] == record.get(k) for k in ("intra_op_threads", "inter_op_threads", "spinning",
                                                   "graph_optimisation"))
        if not same:
            problems.append(f"runtime {rid}: differs from the runtime block of {r['sources'][0]['file']}")
    measured = {}
    for s in label["speed"]:
        if s["status"] != "measured":
            continue
        threads = s["settings"]["threads"]
        where = f"speed {s['build_id']} {threads} threads"
        path = Path(s["sources"][0]["file"])
        record = json.loads(path.read_text(encoding="utf-8"))
        m, st, fp = record["metrics"], record["settings"], record["device"]["fingerprint"]
        physical = sorted({st["physical_core_of_each_cpu"][str(c)] for c in st["pinned_cpus"]})
        with np.load(path.with_name(record["arrays"]["file"])) as data:
            window = {"first_start": utc(float(data["run_starts"].min())),
                      "last_end": utc(float(data["run_ends"].max()))}
        expected = {
            "p50_ms": m["p50_ms"]["value"], "p95_ms": m["p95_ms"]["value"], "p99_ms": m["p99_ms"]["value"],
            "spread_pct": m["spread_pct"]["value"], "unstable": st["unstable"],
            "unstable_above_pct": st["unstable_above_pct"], "sessions": st["sessions"],
            "warmup_runs": st["warmup_runs"], "timed_runs": st["timed_runs"],
            "discarded_sessions": len(st["discarded_sessions"]), "vnni": st["vnni"]["text"],
            "timed_utc": window,
            "pinning": {"logical_cpus": st["pinned_cpus"], "physical_cores": physical,
                        "n_logical": len(st["pinned_cpus"]), "n_physical": len(physical),
                        "core_kind": "performance", "reported_by": fp["os"], "read_back": False},
        }
        for key, value in expected.items():
            if s.get(key) != value:
                problems.append(f"{where}: {key} differs from {path}")
        if st["pinned_cpus"] != fp["core_types"]["performance"]:
            problems.append(f"{where}: {path} was not pinned to the performance cores")
        if threads != record["runtime"]["threads"] or record["code"]["dirty"]:
            problems.append(f"{where}: thread count differs, or {path} is from uncommitted code")
        if record["derived_from"][0]["sha256"] != builds_by_id[s["build_id"]]["file"]["sha256"]:
            problems.append(f"{where}: {path} timed another file than this build")
        rt, own = runtimes.get(s["runtime_id"]), record["runtime"]
        fields = ("name", "version", "execution_provider", "threads", "intra_op_threads", "inter_op_threads",
                  "spinning", "graph_optimisation")
        if rt is None or any(rt[k] != own.get(k) for k in fields):
            problems.append(f"{where}: its runtime is not the one its latency record states")
        measured[(builds_by_id[s["build_id"]]["role"], threads)] = s
    for (role, threads), s in measured.items():
        if role != "labelled":
            continue
        ratio = s["p50_ms"] / measured[("reference", threads)]["p50_ms"]
        ratio_ok = s.get("time_vs_reference", {}).get("ratio_p50") == ratio
        if not ratio_ok or s["time_vs_reference"]["slower"] != (ratio > 1):
            problems.append(f"speed {s['build_id']} {threads} threads: time vs FP32 differs from the records")
    return problems


def check_licences(label: dict) -> list:
    """Step 8: the licence section against the build records."""
    lic = label.get("licences") or {}
    named = [json.loads(Path(s["file"]).read_text(encoding="utf-8")) for s in label["sources"]
             if s["file"].startswith("models/") and s["file"].endswith(".json")]
    records = {b["role"]: next(r for r in named if r["file"]["sha256"] == b["file"]["sha256"])
               for b in label["builds"]}
    trained_on = "ImageNet-1k" if "IMAGENET1K" in records["reference"]["weights"].upper() else None
    expected = {
        "brokkr_code": "Apache-2.0",
        "model_code": records["labelled"]["licence"]["code"],
        "model_weights": records["labelled"]["licence"]["weights"],
        "weights_trained_on": trained_on,
        "label_data": label["generated"]["label_licence"],
    }
    problems = [f"licences: {k} differs from the build records" for k, v in expected.items()
                if lic.get(k) != v]
    if records["reference"]["licence"] != records["labelled"]["licence"]:
        problems.append("licences: the two build records state different licences")
    if lic.get("label_data") != "CC BY 4.0" or not all(lic.get(k) for k in ("model_code", "model_weights")):
        problems.append("licences: the section is incomplete")
    return problems


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
        elif b["recipe"].get("skip_symbolic_shape", False) != (match[0].get("settings") or {}).get(
            "skip_symbolic_shape", False
        ):
            problems.append(f"build {b['build_id']}: skip_symbolic_shape differs from its build record")
        elif b["failure"]:
            f, sanity = b["failure"], match[0]["sanity_check"]
            expected = (sanity["top1_agreement_with_fp32"], MIN_AGREEMENT_WITH_FP32, sanity["n_images"],
                        sanity["split"])
            if (f["value"], f["limit"], f["n_items"], f["split"]) != expected:
                problems.append(f"build {b['build_id']}: failure differs from its build record")
    entry = load_model_list()[label["model"]["name"]]
    names = load_precision_display_names()
    key = {b["build_id"]: "fp32" if b["precision"] == "fp32" else f"int8_{b['recipe']['method']}"
           for b in label["builds"]}
    if label["model"].get("display_name") != entry["display_name"] or any(
        b.get("display_name") != names[key[b["build_id"]]] for b in label["builds"]
    ):
        problems.append("a display name differs from brokkr_edge/model_list.json")

    # 4 and 5. Every measurement, and the runtime of the accuracy record it comes from.
    record_runtime = {}
    for m in label["measurements"]:
        if m["metric"] == "top1":
            rt = json.loads(Path(m["sources"][0]["file"]).read_text(encoding="utf-8"))["runtime"]
            match = [r["runtime_id"] for r in label["runtimes"]
                     if all(r[k] == rt.get(k) for k in ("name", "version", "execution_provider", "threads",
                                                        "spinning"))]
            record_runtime[(m["build_id"], m["condition_id"])] = match[0] if match else None
    reference_id = next(b["build_id"] for b in label["builds"] if b["role"] == "reference")
    for m in label["measurements"]:
        where = f"{m['metric']} {m['build_id']} {m['condition_id']}"
        if m["runtime_id"] != record_runtime.get((m["build_id"], m["condition_id"])):
            problems.append(f"{where}: runtime differs from its record")
        other = {"damage_drop": (m["build_id"], "clean"),
                 "shrinking_cost": (reference_id, m["condition_id"])}.get(m["metric"])
        if other:
            paired_rt = record_runtime.get(other)
            expected = paired_rt if paired_rt != m["runtime_id"] else None
            if m["settings"].get("paired_runtime_id") != expected:
                problems.append(f"{where}: paired_runtime_id differs from its second record")
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
            expected = ("INT8 build failed", [], [])
        else:
            coverage_ci, drop_ci = ix[("coverage", bid, cid)]["ci95"], ix[("damage_drop", bid, cid)]["ci95"]
            state, _ = envelope_state(coverage_ci, drop_ci, rule)
            flags = []
            if build["role"] == "labelled":
                fp32 = ix[("top1", builds["reference"]["build_id"], cid)]["value"]
                flags = shrinking_cost_flags(ix[("shrinking_cost", bid, cid)]["ci95"], fp32, rule)
            expected = (state, failed_lines(coverage_ci, drop_ci, rule), flags)
        found = (row["state"], row["failed"], row["shrinking_cost_flags"])
        if found != expected:
            problems.append(f"envelope {bid} {cid}: {found}, rule gives {expected}")
    if summary(label["envelope"]["rows"], label["label_id"]) != label["summary"]:
        problems.append("summary differs from the one the envelope gives")
    sanity = label["checks"]["fp32_sanity"]
    measured = ix[("top1", builds["reference"]["build_id"], "clean")]["value"]
    published = load_model_list()[label["model"]["name"]]["published"]["top1"]
    if (sanity["measured"], sanity["published"]) != (measured, published) or sanity["pass"] != (
        abs(measured - published) <= sanity["tolerance"]
    ):
        problems.append("FP32 sanity check differs from its sources")

    # 7 and 8. Speed, runtimes and licences.
    problems += check_speed(label, {b["build_id"]: b for b in label["builds"]})
    problems += check_licences(label)

    # 9. Renders.
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
    card = ModelCard(md).data.to_dict()
    licence_meta = (card.get("license"), card.get("license_link"))
    if licence_meta != ("other", "#licences") or "\n## Licences\n" not in md:
        problems.append("the model card's licence metadata is not as expected, or its link has no section")
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
readme = subprocess.run(
    [sys.executable, "scripts/41_readme_label_example.py", "--check", "--labels", args.labels],
    capture_output=True, text=True,
)
print((readme.stdout + readme.stderr).strip())
all_ok &= readme.returncode == 0
print(f"\n{'PASS' if all_ok else 'FAIL'}: {len(folders)} labels in {args.labels}")
sys.exit(0 if all_ok else 1)
