"""Make the P1 labels from the 4.1 records (no model is run).

Usage:  python scripts/39_make_labels.py [--models mobilenet_v3_large mobilenet_v3_small]
                                         [--out labels]
Needs:  the 4.1 test and conformal_calibration records with their scores (results/breadth), their
        reliability records (results/breadth_reliability), the build records (models/), and
        brokkr_edge/model_list.json
Writes: <out>/<model>/label.json (the only source), label.md (Hugging Face model card) and label.html

Rules (docs/label_schema.md; docs/hypotheses_stage4.md, notes of 29-30 September 2026):
- Every input record must pass the schema check and come from a clean commit, or the label is refused.
- One label per shrunk build (the 4.1 Percentile INT8), with FP32 beside it. A failed INT8 build gives
  a label whose INT8 rows say "INT8 build failed", with the reason from its build record.
- Every number is copied from a record field, or computed from saved scores (damage drop, shrinking
  cost: paired bootstrap, 1,000 resamples, seed 0), and names its source file, checksum and field.
- Speed rows say "not measured" until the P1 latency run; the Raspberry Pi 5 row until a Pi exists.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from brokkr_edge import __version__
from brokkr_edge.fingerprint import git_info
from brokkr_edge.judge import top1_correct
from brokkr_edge.label import (
    CI_LEVEL,
    ENVELOPE_RULE,
    envelope_state,
    failed_lines,
    paired,
    shrinking_cost_flag,
    summary,
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
    condition_label,
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


def write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def make_label(model: str) -> dict:
    used_files.clear()
    entry = load_model_list()[model]
    note(MODEL_LIST_FILE, "generated from torchvision metadata; tests/test_model_list.py")
    mid = model_id(entry["publisher"], model, entry["weights"])

    # Builds.
    builds, status = {}, {}
    for role, precision in PRECISIONS.items():
        path = Path("models") / f"{model}_{precision}.json"
        build = json.loads(path.read_text(encoding="utf-8"))
        state, problems = check_build_record(build, set())
        if problems:
            sys.exit(f"REFUSED: {path}: {problems}")
        note(path, "brokkr_edge.schema.check_build_record")
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
            }
            if precision != "fp32"
            else {"method": "fp32 export (torch.onnx), not quantized"}
        )
        failure = None
        if state == "failed":
            failed = [name for name, ok in build["build"]["checks"].items() if not ok]
            failure = {
                "check": failed[0],
                "value": build["sanity_check"]["top1_agreement_with_fp32"],
                "limit": MIN_AGREEMENT_WITH_FP32,
                "n_items": build["sanity_check"]["n_images"],
                "split": build["sanity_check"]["split"],
                "sources": [
                    source(path, "sanity_check.top1_agreement_with_fp32"),
                    source(path, "sanity_check.n_images"),
                    source(path, "sanity_check.split"),
                ],
            }
        bid = make_build_id(mid, "fp32" if precision == "fp32" else "int8", build["file"]["sha256"])
        builds[role] = {
            "build_id": bid,
            "role": role,
            "display_name": load_precision_display_names()[precision],
            "precision": "fp32" if precision == "fp32" else "int8",
            "recipe": recipe,
            "file": {"sha256": build["file"]["sha256"], "size_bytes": build["file"]["size_bytes"]},
            "status": state,
            "failure": failure,
        }
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
    runtime = ref_clean["runtime"]
    hardware = [
        {
            "hardware_id": hw_id,
            "kind": "laptop",
            "cpu_model": fp["cpu_model"],
            "os": f"{fp['os']} {fp['os_release']}",
            "architecture": fp.get("architecture"),
            "cores": {"types": fp.get("core_types"), "map": None},
            "features": (fp.get("cpu_features") or {}).get("features"),
            "runtime": {
                "name": runtime["name"],
                "version": runtime["version"],
                "execution_provider": runtime["execution_provider"],
                "threads": runtime["threads"],
            },
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
    conditions = [
        {
            "condition_id": condition_id(c["suite"], c["corruption"], c["severity"]),
            "suite": c["suite"],
            "damage": c["corruption"],
            "severity": c["severity"] or None,
            "modality": "image",
            "label": condition_label(c),
        }
        for c in CONDITIONS
    ]

    # Measurements.
    measurements = []

    def add(metric, role, cid, value, ci, n, settings, sources):
        measurements.append(
            {
                "metric": metric,
                "build_id": builds[role]["build_id"],
                "dataset_id": test_id,
                "condition_id": cid,
                "hardware_id": hw_id,
                "value": value,
                "ci95": ci,
                "n_items": n,
                "unit": "classes" if metric == "mean_set_size" else "fraction",
                "settings": settings,
                "sources": sources,
            }
        )

    details_conformal = {}
    for (role, cid), (path, record) in acc.items():
        n = record["data"]["n_images"]
        for metric in ("top1", "top5"):
            m = record["metrics"][metric]
            add(metric, role, cid, m["value"], m["ci95"], n, {}, [source(path, f"metrics.{metric}")])
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
            add(metric, role, cid, m["value"], m["ci95"], n, settings, [source(rpath, f"metrics.{field}")])
    for role in PRECISIONS:
        if status[role] != "usable":
            continue
        clean_correct = correct[(role, "clean")][0]
        for c in conditions[1:]:
            cid = c["condition_id"]
            p = paired(correct[(role, cid)][0], clean_correct)
            add(
                "damage_drop",
                role,
                cid,
                p["value"],
                p["ci95"],
                p["n_items"],
                {"paired_with": "clean", "bootstrap": p["bootstrap"]},
                [
                    source(acc[(role, cid)][0].with_suffix(".npz"), "logits, labels"),
                    source(acc[(role, "clean")][0].with_suffix(".npz"), "logits, labels"),
                ],
            )
    if status["labelled"] == "usable":
        for c in conditions:
            cid = c["condition_id"]
            p = paired(correct[("labelled", cid)][0], correct[("reference", cid)][0])
            add(
                "shrinking_cost",
                "labelled",
                cid,
                p["value"],
                p["ci95"],
                p["n_items"],
                {"paired_with": builds["reference"]["build_id"], "bootstrap": p["bootstrap"]},
                [
                    source(acc[("labelled", cid)][0].with_suffix(".npz"), "logits, labels"),
                    source(acc[("reference", cid)][0].with_suffix(".npz"), "logits, labels"),
                ],
            )

    # Envelope and summary.
    index = {(m["metric"], m["build_id"], m["condition_id"]): m for m in measurements}
    rows = []
    for role in ("reference", "labelled"):
        bid = builds[role]["build_id"]
        for c in conditions[1:]:
            cid = c["condition_id"]
            if status[role] != "usable":
                f = builds[role]["failure"]
                rows.append(
                    {
                        "build_id": bid,
                        "condition_id": cid,
                        "hardware_id": hw_id,
                        "state": "INT8 build failed",
                        "why": [f"build check failed: {f['check']}"],
                        "failed": [],
                        "shrinking_cost_flag": None,
                    }
                )
                continue
            coverage_ci = index[("coverage", bid, cid)]["ci95"]
            drop_ci = index[("damage_drop", bid, cid)]["ci95"]
            state, why = envelope_state(coverage_ci, drop_ci)
            flag = None
            if role == "labelled":
                fp32_top1 = index[("top1", builds["reference"]["build_id"], cid)]["value"]
                flag = shrinking_cost_flag(index[("shrinking_cost", bid, cid)]["ci95"], fp32_top1)
            rows.append(
                {
                    "build_id": bid,
                    "condition_id": cid,
                    "hardware_id": hw_id,
                    "state": state,
                    "why": why,
                    "failed": failed_lines(coverage_ci, drop_ci),
                    "shrinking_cost_flag": flag,
                }
            )

    # Speed: not measured yet.
    speed = []
    for role in ("reference", "labelled"):
        reason_laptop = (
            "INT8 build failed"
            if status[role] != "usable"
            else "the P1 laptop latency run has not been done yet"
        )
        for threads in (1, 4):
            speed.append(
                {
                    "build_id": builds[role]["build_id"],
                    "hardware_id": hw_id,
                    "hardware_kind": "laptop",
                    "status": "not measured",
                    "reason": reason_laptop,
                    "settings": {"threads": threads},
                }
            )
        speed.append(
            {
                "build_id": builds[role]["build_id"],
                "hardware_id": None,
                "hardware_kind": "raspberry-pi-5",
                "status": "not measured",
                "reason": "no Raspberry Pi 5 yet (Platform plan step 6)",
                "settings": {},
            }
        )

    measured_fp32 = index[("top1", builds["reference"]["build_id"], "clean")]["value"]
    published = entry["published"]["top1"]
    n_damaged = len(conditions) - 1
    label = {
        "schema_version": 1,
        "label_id": builds["labelled"]["build_id"],
        "source": {
            "kind": "official",
            "verified": True,
            "submitted_by": None,
            "how_made": "scripts/39_make_labels.py, from the 4.1 records",
        },
        "generated": {
            "by": f"brokkr-edge {__version__}",
            "commit": git["commit"],
            "dirty": git["dirty"],
            "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "ci_level": CI_LEVEL,
            "label_licence": "CC BY 4.0",
            "envelope_rule": ENVELOPE_RULE["name"],
        },
        "model": {
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
        "builds": [builds["reference"], builds["labelled"]],
        "hardware": hardware,
        "datasets": datasets,
        "conditions": conditions,
        "checks": {
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
        "measurements": measurements,
        "envelope": {"rule": ENVELOPE_RULE, "rows": rows},
        "summary": summary(rows, builds["labelled"]["build_id"]),
        "speed": speed,
        "details": {"conformal": details_conformal},
        "limits": [
            "The damage is simulated (Brokkr's own and ImageNet-C corruptions); real fog, darkness "
            "or noise may "
            "affect the model differently.",
            "The ImageNet-C conditions were made with the official corruption code on these test "
            "images, with "
            "fixed seeds and each model's own preprocessing, so they are not directly comparable to "
            "published "
            "ImageNet-C results.",
            f"Measured on one machine: {fp['cpu_model']}, {fp['os']} {fp['os_release']}.",
            f"With {n_damaged} conditions, an occasional result may cross a line by chance.",
            "Coverage is for prediction sets tuned on clean calibration images; under damage there is no "
            "coverage promise, and the label shows what was measured.",
            "Accuracy was measured on the laptop; it has not been checked on other hardware.",
        ],
        "sources": [
            {"file": f, "sha256": sha, "check": check} for f, (sha, check) in sorted(used_files.items())
        ],
    }
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
    print(f"     large shrinking cost: {', '.join(label['summary']['large_shrinking_cost']) or 'none'}")
print("PASS" if all_ok else "FAIL")
sys.exit(0 if all_ok else 1)
