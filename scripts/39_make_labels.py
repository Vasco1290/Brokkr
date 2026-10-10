"""Make the P1 labels from the 4.1 records and the laptop latency records (no model is run).

Usage:  python scripts/39_make_labels.py [--models mobilenet_v3_large mobilenet_v3_small]
                                         [--out labels]
Needs:  the 4.1 test and conformal_calibration records with their scores (results/breadth), their
        reliability records (results/breadth_reliability), the laptop latency records with their timings
        (results/latency), the build records (models/), and brokkr_edge/model_list.json
Writes: <out>/<model>/label.json (the only source), label.md (Hugging Face model card) and label.html

Rules (docs/label_schema.md; docs/hypotheses_stage4.md, notes of 29 September to 3 October 2026):
- Every input record must pass the schema check and come from a clean commit, or the label is refused.
- One label per shrunk build (the 4.1 Percentile INT8), with FP32 beside it. A failed INT8 build gives
  a label whose INT8 rows say "INT8 build failed", with the reason from its build record.
- Every number is copied from a record field, or computed from saved scores (damage drop, shrinking
  cost: paired bootstrap, 1,000 resamples, seed 0) or from two copied numbers (INT8's p50 over FP32's),
  and names its source file, checksum and field.
- Laptop speed rows come from the latency records; the Raspberry Pi 5 rows say "not measured".
- The licence section comes from the build records (note of 3 October 2026, point 6).

This script finds, checks and reads the study's records; the label is put together by brokkr_edge.label_build,
which the user path shares (docs/user_models.md, note of 10 October 2026).
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from brokkr_edge.fingerprint import git_info
from brokkr_edge.judge import top1_correct
from brokkr_edge.label_build import (
    STUDY_MACHINE_SENTENCE,
    assemble,
    build_entry,
    conditions_block,
    derived_measurements,
    envelope_rows,
    failure_block,
    general_limits,
    generated_block,
    measurement,
    runtime_entry,
)
from brokkr_edge.label_render import to_html, to_markdown, unexplained_numbers
from brokkr_edge.label_schema import build_id as make_build_id
from brokkr_edge.label_schema import (
    check_label,
    condition_id,
    dataset_id,
    hardware_id,
    model_id,
)
from brokkr_edge.model_list import MODEL_LIST_FILE, load_model_list, load_precision_display_names
from brokkr_edge.results import sha256_of
from brokkr_edge.schema import (
    MIN_AGREEMENT_WITH_FP32,
    check_build_record,
    check_record,
    condition,
    load_measurement,
)

DATASET = "imagenet-1k-val"
PRECISIONS = {"reference": "fp32", "labelled": "int8_percentile99.99"}
CONDITIONS = (
    [condition()]
    + [
        condition("fog", "brokkr", 3),
        condition("darkness", "brokkr", 5),
        condition("defocus_blur", "brokkr", 3),
        condition("noise", "brokkr", 3),
    ]
    + [
        condition(c, "imagenet-c", s)
        for c in ("fog", "contrast", "defocus_blur", "gaussian_noise")
        for s in (3, 5)
    ]
)
SANITY_TOLERANCE = 0.010  # FP32 clean top-1 within 1.0 point of torchvision's published top-1 (4.1 rule)
THREAD_COUNTS = (1, 4)  # the laptop latency method's thread counts (note of 29 September 2026)
BROKKR_CODE_LICENCE = "Apache-2.0"
LABEL_DATA_LICENCE = "CC BY 4.0"
RELIABILITY = {
    "coverage": ("conformal", "coverage"),
    "mean_set_size": ("conformal", "mean_set_size"),
    "ece": ("calibration", "ece"),
    "e_aurc": ("selective", "e_aurc"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=list(load_model_list()))
parser.add_argument("--out", default="labels")
args = parser.parse_args()
git = git_info()
used_files = {}  # path -> (sha256, what checked it)


def repo_path(path: Path) -> str:
    """The path from the repository folder, so a label never shows this machine's folders."""
    return (path.relative_to(Path.cwd()) if path.is_absolute() else path).as_posix()


def note(path: Path, check: str) -> str:
    sha = sha256_of(path)
    used_files[repo_path(path)] = (sha, check)
    return sha


def source(path: Path, field: str) -> dict:
    return {"file": repo_path(path), "sha256": used_files[repo_path(path)][0], "field": field}


def checked(path: Path) -> tuple:
    """A schema-2 record and its scores, refused unless it passes the check and is from a clean commit."""
    record, arrays = (
        load_measurement(path)
        if json.loads(path.read_text(encoding="utf-8")).get("arrays")
        else (json.loads(path.read_text(encoding="utf-8")), None)
    )
    problems = check_record(record) + (["made from uncommitted code"] if record["code"]["dirty"] else [])
    if problems:
        sys.exit(f"REFUSED: {path} fails the check: {problems}")
    note(path, "brokkr_edge.schema.check_record (the rules of scripts/22_check_results.py), clean commit")
    if arrays is not None:
        note(path.with_suffix(".npz"), "checksum equal to its record's")
    return record, arrays


def record_runtime(record: dict, path: Path) -> dict:
    """The `runtimes` entry of a checked record."""
    return runtime_entry(record["runtime"], [source(path, "runtime")])


def utc(seconds: float) -> str:
    return datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc).isoformat(timespec="seconds")


def write(path: Path, text: str) -> None:
    """Write `text` with LF line endings on every OS (the label files are then identical everywhere)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def make_label(model: str) -> dict:
    used_files.clear()
    entry = load_model_list()[model]
    note(MODEL_LIST_FILE, "generated from torchvision metadata; tests/test_model_list.py")
    mid = model_id(entry["publisher"], model, entry["weights"])

    # Builds.
    builds, status, build_records = {}, {}, {}
    for role, precision in PRECISIONS.items():
        path = Path("models") / f"{model}_{precision}.json"
        build = json.loads(path.read_text(encoding="utf-8"))
        state, problems = check_build_record(build, set())
        if problems:
            sys.exit(f"REFUSED: {path}: {problems}")
        note(path, "brokkr_edge.schema.check_build_record")
        build_records[role] = (path, build)
        onnx_file = Path("models") / f"{model}_{precision}.onnx"
        if sha256_of(onnx_file) != build["file"]["sha256"]:
            sys.exit(f"REFUSED: {onnx_file} does not match its build record")
        s = build.get("settings") or {}
        recipe = (
            {
                "method": s["calibration_method"],
                "detail": s.get("calibration_method_detail"),
                "format": s.get("format"),
                "weights": s.get("weights"),
                "activations": s.get("activations"),
                "calibration": {
                    "dataset_id": dataset_id("imagenet", DATASET, s["calibration"]["split"]),
                    "n_items": s["calibration"]["n_images"],
                    "group_items": s["calibration"].get("group_images"),
                },
                "kept_float": s.get("kept_float"),
                "skip_symbolic_shape": s.get("skip_symbolic_shape", False),
            }
            if precision != "fp32"
            else {"method": "fp32 export (torch.onnx), not quantized"}
        )
        failure = None
        if state == "failed":
            failed = [name for name, ok in build["build"]["checks"].items() if not ok]
            failure = failure_block(
                failed[0],
                build["sanity_check"]["top1_agreement_with_fp32"],
                MIN_AGREEMENT_WITH_FP32,
                build["sanity_check"]["n_images"],
                build["sanity_check"]["split"],
                [
                    source(path, "sanity_check.top1_agreement_with_fp32"),
                    source(path, "sanity_check.n_images"),
                    source(path, "sanity_check.split"),
                ],
            )
        bid = make_build_id(mid, "fp32" if precision == "fp32" else "int8", build["file"]["sha256"])
        builds[role] = build_entry(
            bid, role, load_precision_display_names()[precision], "fp32" if precision == "fp32" else "int8",
            recipe, build["file"]["sha256"], build["file"]["size_bytes"], state, failure,
        )
        status[role] = state

    # Records: accuracy (with scores) and reliability, for every usable build and condition.
    acc, rel, correct = {}, {}, {}
    for role, precision in PRECISIONS.items():
        if status[role] != "usable":
            continue
        for cond in CONDITIONS:
            suite = cond["suite"] or "none"
            stem = f"{model}_{precision}_{DATASET}_test_{suite}_{cond['corruption']}_s{cond['severity']}"
            path = Path("results/breadth") / f"{stem}.json"
            record, arrays = checked(path)
            cid = condition_id(cond["suite"], cond["corruption"], cond["severity"])
            acc[(role, cid)] = (path, record)
            correct[(role, cid)] = (top1_correct(arrays["logits"], arrays["labels"]), arrays["positions"])
            for kind in ("conformal", "calibration", "selective"):
                rpath = Path("results/breadth_reliability") / f"{stem}_{kind}.json"
                rel[(role, cid, kind)] = (rpath, checked(rpath)[0])
    ref_clean_path, ref_clean = acc[("reference", "clean")]
    cal_path = Path("results/breadth") / f"{model}_fp32_{DATASET}_conformal_calibration_none_clean_s0.json"
    cal_record, cal_arrays = checked(cal_path)

    # One machine, one runtime, the same images everywhere.
    fp = ref_clean["device"]["fingerprint"]
    for path, record in list(acc.values()) + [(p, r) for p, r in rel.values()]:
        f = record["device"]["fingerprint"]
        if (f["cpu_model"], f["os"], f["os_release"]) != (fp["cpu_model"], fp["os"], fp["os_release"]):
            sys.exit(f"REFUSED: {path} was measured on another machine")
    positions = {tuple(p.tolist()) for _, p in correct.values()}
    if len(positions) != 1:
        sys.exit("REFUSED: the records do not hold the same test images")
    test_positions = set(next(iter(positions)))
    if test_positions & set(cal_arrays["positions"].tolist()):
        sys.exit("REFUSED: calibration and test images overlap")
    hw_id = hardware_id("laptop", fp["cpu_model"], f"{fp['os']} {fp['os_release']}")
    runtimes, acc_runtime = [], {}  # each accuracy record's runtime (note of 3 October 2026)
    for key, (path, record) in acc.items():
        rt = record_runtime(record, path)
        if rt["runtime_id"] not in {r["runtime_id"] for r in runtimes}:
            runtimes.append(rt)
        acc_runtime[key] = rt["runtime_id"]
    hardware = [
        {
            "hardware_id": hw_id,
            "kind": "laptop",
            "cpu_model": fp["cpu_model"],
            "os": f"{fp['os']} {fp['os_release']}",
            "architecture": fp.get("architecture"),
            "cores": {"types": fp.get("core_types"), "map": None},
            "features": (fp.get("cpu_features") or {}).get("features"),
            "fingerprint_source": source(ref_clean_path, "device.fingerprint"),
        }
    ]

    test_id = dataset_id("imagenet", DATASET, "test")
    cal_id = dataset_id("imagenet", DATASET, "conformal_calibration")
    licence = ref_clean["data"]["licence"]
    datasets = [
        {
            "dataset_id": test_id,
            "name": "ImageNet-1k validation",
            "split": "test",
            "n_items": ref_clean["data"]["n_images"],
            "item_kind": "image",
            "licence": licence,
            "disjoint_from": [cal_id],
        },
        {
            "dataset_id": cal_id,
            "name": "ImageNet-1k validation",
            "split": "conformal_calibration",
            "n_items": cal_record["data"]["n_images"],
            "item_kind": "image",
            "licence": licence,
            "disjoint_from": [test_id],
        },
    ]
    conditions = conditions_block(CONDITIONS)
    condition_ids = [c["condition_id"] for c in conditions]

    # Measurements: copied from the records, then the derived ones (label_build).
    measurements = []
    details_conformal = {}
    for (role, cid), (path, record) in acc.items():
        n = record["data"]["n_images"]

        def add(metric, value, ci, settings, sources, role=role, cid=cid, n=n):
            measurements.append(measurement(metric, builds[role]["build_id"], test_id, cid, hw_id,
                                            acc_runtime[(role, cid)], value, ci, n, settings, sources))

        for metric in ("top1", "top5"):
            m = record["metrics"][metric]
            add(metric, m["value"], m["ci95"], {}, [source(path, f"metrics.{metric}")])
        for metric, (kind, field) in RELIABILITY.items():
            rpath, rrec = rel[(role, cid, kind)]
            m = rrec["metrics"][field]
            settings = {}
            if kind == "conformal":
                settings = {
                    "target_coverage": rrec["settings"]["target_coverage"],
                    "threshold": rrec["settings"]["threshold"],
                    "calibration_dataset_id": cal_id,
                    "calibration_items": rrec["settings"]["calibration_images"],
                    "method": "LAC",
                }
                details_conformal[builds[role]["build_id"]] = {
                    "target": settings["target_coverage"],
                    "threshold": settings["threshold"],
                    "calibration_items": settings["calibration_items"],
                }
            add(metric, m["value"], m["ci95"], settings, [source(rpath, f"metrics.{field}")])
    measurements += derived_measurements(
        builds, status, condition_ids,
        {key: c for key, (c, _) in correct.items()},
        {key: source(path.with_suffix(".npz"), "logits, labels") for key, (path, _) in acc.items()},
        acc_runtime, test_id, hw_id,
    )

    # Envelope (the summary is made from it by label_build.assemble).
    rows = envelope_rows(builds, status, condition_ids[1:], measurements, hw_id)

    # Speed: laptop rows from the latency records (note of 3 October 2026), Raspberry Pi 5 not measured.
    speed, latency = [], {}
    for role, precision in PRECISIONS.items():
        for threads in THREAD_COUNTS:
            if status[role] != "usable":
                speed.append({"build_id": builds[role]["build_id"], "hardware_id": hw_id,
                              "hardware_kind": "laptop", "runtime_id": None, "status": "not measured",
                              "reason": "INT8 build failed", "settings": {"threads": threads}})
                continue
            path = Path("results/latency") / f"{model}_{precision}_laptop_{threads}threads.json"
            record, arrays = checked(path)
            f, s = record["device"]["fingerprint"], record["settings"]
            if (f["cpu_model"], f["os"], f["os_release"]) != (fp["cpu_model"], fp["os"], fp["os_release"]):
                sys.exit(f"REFUSED: {path} was timed on another machine")
            if record.get("derived_from", [{}])[0].get("sha256") != builds[role]["file"]["sha256"]:
                sys.exit(f"REFUSED: {path} timed another file than the labelled build")
            if s["pinned_cpus"] != f["core_types"]["performance"]:
                sys.exit(f"REFUSED: {path} was not pinned to the performance cores its fingerprint lists")
            rt = record_runtime(record, path)
            if rt["runtime_id"] not in {r["runtime_id"] for r in runtimes}:
                runtimes.append(rt)
            physical = sorted({s["physical_core_of_each_cpu"][str(c)] for c in s["pinned_cpus"]})
            m = record["metrics"]
            row = {
                "build_id": builds[role]["build_id"],
                "hardware_id": hw_id,
                "hardware_kind": "laptop",
                "runtime_id": rt["runtime_id"],
                "status": "measured",
                "reason": None,
                "settings": {"threads": threads, "batch": s["batch"], "input": s["input"],
                             "what_is_timed": s["what_is_timed"]},
                "p50_ms": m["p50_ms"]["value"],
                "p95_ms": m["p95_ms"]["value"],
                "p99_ms": m["p99_ms"]["value"],
                "spread_pct": m["spread_pct"]["value"],
                "unstable": s["unstable"],
                "unstable_above_pct": s["unstable_above_pct"],
                "sessions": s["sessions"],
                "warmup_runs": s["warmup_runs"],
                "timed_runs": s["timed_runs"],
                "discarded_sessions": len(s["discarded_sessions"]),
                "timed_utc": {"first_start": utc(float(arrays["run_starts"].min())),
                              "last_end": utc(float(arrays["run_ends"].max()))},
                "pinning": {"logical_cpus": s["pinned_cpus"], "physical_cores": physical,
                            "n_logical": len(s["pinned_cpus"]), "n_physical": len(physical),
                            "core_kind": "performance", "reported_by": f["os"], "read_back": False},
                "vnni": s["vnni"]["text"],
                "sources": [source(path, f"metrics.{k}.value")
                            for k in ("p50_ms", "p95_ms", "p99_ms", "spread_pct")]
                + [source(path, k) for k in ("settings.unstable", "settings.unstable_above_pct",
                                             "settings.pinned_cpus", "settings.physical_core_of_each_cpu",
                                             "settings.discarded_sessions")]
                + [source(path.with_suffix(".npz"), "run_starts, run_ends")],
            }
            latency[(role, threads)] = (path, row)
            speed.append(row)
    for threads in THREAD_COUNTS:  # INT8's time as a multiple of FP32's, same machine and thread count
        if ("labelled", threads) in latency:
            ref_path, ref_row = latency[("reference", threads)]
            lab_path, lab_row = latency[("labelled", threads)]
            ratio = lab_row["p50_ms"] / ref_row["p50_ms"]
            lab_row["time_vs_reference"] = {
                "ratio_p50": ratio,
                "reference_p50_ms": ref_row["p50_ms"],
                "slower": ratio > 1,
                "sources": [source(p, "metrics.p50_ms.value") for p in (lab_path, ref_path)],
            }
    timed = [r for _, r in latency.values()]
    if len({(r["sessions"], r["warmup_runs"], r["timed_runs"], r["unstable_above_pct"]) for r in timed}) > 1:
        sys.exit(f"REFUSED: {model}'s latency records differ in sessions, runs or the unstable line")
    for role in PRECISIONS:
        speed.append(
            {
                "build_id": builds[role]["build_id"],
                "hardware_id": None,
                "hardware_kind": "raspberry-pi-5",
                "runtime_id": None,
                "status": "not measured",
                "reason": "no Raspberry Pi 5 yet (Platform plan step 6)",
                "settings": {},
            }
        )

    # Licences, from the build records (note of 3 October 2026, point 6).
    ref_build_path, ref_build = build_records["reference"]
    lab_build_path, lab_build = build_records["labelled"]
    if ref_build["licence"] != lab_build["licence"]:
        sys.exit(f"REFUSED: {model}'s build records state different licences")
    licences = {
        "brokkr_code": BROKKR_CODE_LICENCE,
        "model_code": lab_build["licence"]["code"],
        "model_weights": lab_build["licence"]["weights"],
        "weights_trained_on": "ImageNet-1k" if "IMAGENET1K" in ref_build["weights"].upper() else None,
        "label_data": LABEL_DATA_LICENCE,
        "sources": [source(lab_build_path, "licence.code"), source(lab_build_path, "licence.weights"),
                    source(ref_build_path, "licence"), source(ref_build_path, "weights")],
    }

    measured_fp32 = next(m["value"] for m in measurements if (m["metric"], m["build_id"], m["condition_id"])
                         == ("top1", builds["reference"]["build_id"], "clean"))
    published = entry["published"]["top1"]
    limits = general_limits(fp["cpu_model"], f"{fp['os']} {fp['os_release']}", len(conditions) - 1,
                            STUDY_MACHINE_SENTENCE)
    if any(not r["pinning"]["read_back"] for r in timed):
        limits.append("Latency is laptop latency from one run on one machine; the CPU pin was not read back "
                      "after it was set.")
    label = assemble(
        source={
            "kind": "official",
            "verified": True,
            "submitted_by": None,
            "how_made": "scripts/39_make_labels.py, from the 4.1 records",
        },
        generated=generated_block(git["commit"], git["dirty"], LABEL_DATA_LICENCE),  # = licences.label_data
        model={
            "model_id": mid,
            "name": model,
            "display_name": entry["display_name"],
            "publisher": entry["publisher"],
            "weights": entry["weights"],
            "task": entry["task"],
            "modality": entry["modality"],
            "input": entry["input"],
            "outputs": entry["outputs"],
            "licence": entry["licence"],
        },
        builds=builds,
        hardware=hardware,
        runtimes=runtimes,
        datasets=datasets,
        conditions=conditions,
        checks={
            "fp32_sanity": {
                "measured": measured_fp32,
                "published": published,
                "tolerance": SANITY_TOLERANCE,
                "pass": abs(measured_fp32 - published) <= SANITY_TOLERANCE,
                "published_source": "torchvision weights metadata (brokkr_edge/model_list.json)",
                "sources": [
                    source(ref_clean_path, "metrics.top1"),
                    source(MODEL_LIST_FILE, f"models.{model}.published.top1"),
                ],
            }
        },
        measurements=measurements,
        rows=rows,
        speed=speed,
        details={"conformal": details_conformal},
        limits=limits,
        licences=licences,
        sources=[{"file": f, "sha256": sha, "check": check}
                 for f, (sha, check) in sorted(used_files.items())],
    )
    if not label["checks"]["fp32_sanity"]["pass"]:
        sys.exit(f"REFUSED: {model}'s FP32 sanity check fails")
    return label


out_root = Path(args.out)
all_ok = True
for model in args.models:
    label = make_label(model)
    problems = check_label(label)
    md, page = to_markdown(label), to_html(label)
    loose = unexplained_numbers(label, md) + unexplained_numbers(label, page)
    folder = out_root / model
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "label.json", json.dumps(label, indent=2) + "\n")
    write(folder / "label.md", md)
    write(folder / "label.html", page)
    ok = not problems and not loose and not label["generated"]["dirty"]
    all_ok &= ok
    print(
        f"{'ok  ' if ok else 'FAIL'} {model}: {folder}/label.json, .md, .html; schema problems "
        f"{len(problems)}; "
        f"unexplained numbers {len(loose)}; commit {label['generated']['commit'][:7]}"
        f"{' (DIRTY)' if label['generated']['dirty'] else ''}"
    )
    for line in problems + loose:
        print(f"     {line}")
    for line in label["summary"]["lines"]:
        count = "" if line["count"] is None else f" ({line['count']})"
        cause = f", {line['cause']}" if line["cause"] else ""
        print(f"     {line['state']}{cause}{count}: {', '.join(line['conditions'])}")
    for name in ("large_shrinking_cost", "reference_harmful", "coverage_failed"):
        part = label["summary"][name]
        print(f"     {name} ({part['count']}): {', '.join(part['conditions']) or 'none'}")
print("PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
