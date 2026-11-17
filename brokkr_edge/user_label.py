"""The label of a user's own model, made from the run's own files in --out (step 3; docs/user_models.md, plan
of 10 October 2026, section b, H's decisions of 10 October 2026, and the notes of 17 November 2026;
docs/label_schema.md, notes of 6 October and 17 November 2026).

It reads only what the run wrote: run_plan.json (the checks), int8_build.json (Brokkr's INT8 build, when
it made one) and the records in records/ with their scores. Every record is checked (brokkr_edge.schema) and
every score file must match its checksum. Then:

- the reliability numbers, for each build and condition, with the functions and settings of the study's
  scripts/31 (decision 2): ECE with 15 bins, E-AURC, and prediction sets for a 90% target whose threshold
  comes from the same build's clean conformal-calibration part (LAC), coverage always with the average set
  size, each with a 1,000-resample bootstrap 95% interval, seed 0;
- the derived numbers, the envelope and the summary, with the functions the study's labels use
  (brokkr_edge.label_build), so a user label is judged by exactly the same lines;
- the user parts: the submitter's declarations, the warnings, "not measured" speed, and the limits that apply.

Every source is named by its path inside --out, with its SHA-256. The label must pass check_label and every
number on its pages must be a label.json number, or nothing is written.
"""

import json
import math
from pathlib import Path

from brokkr_edge.fingerprint import git_info
from brokkr_edge.judge import top1_correct
from brokkr_edge.label_build import (
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
from brokkr_edge.label_schema import (
    SUPPLIED,
    USER_SEED_SCHEME,
    USER_SPEED_REASON,
    build_id,
    check_label,
    condition_id,
    hardware_id,
)
from brokkr_edge.model_list import load_model_list
from brokkr_edge.quantize import INT8_METHODS
from brokkr_edge.results import sha256_of
from brokkr_edge.schema import check_record, load_measurement
from brokkr_edge.shift.conformal import conformal_threshold, evaluate_sets
from brokkr_edge.shift.reliability import calibration, confidence_and_correct, softmax
from brokkr_edge.shift.selective import selective_prediction
from brokkr_edge.user_build import RECORD_FILE as BUILD_FILE
from brokkr_edge.user_checks import WARN_AGREEMENT_BELOW
from brokkr_edge.user_plan import REPOSITORY
from brokkr_edge.user_run import FP32, INT8_BROKKR, INT8_SUPPLIED, record_path

TARGET_COVERAGE = 0.90
PLAN_FILE = "run_plan.json"
STUDY_SIZE = 224  # the study's input size; other sizes get the D8 sentence
LABEL_DATA = "chosen by the submitter; labels submitted to Brokkr's catalog are CC BY 4.0 (step 4)"
MACHINE_SENTENCE = (
    "Accuracy was measured only on the machine that ran this test; it has not been checked on "
    "other hardware."
)  # decision 6
SEED_SENSITIVITY = REPOSITORY / "results" / "checks" / "seed_sensitivity.json"
RANDOM_DAMAGE = {
    ("brokkr", "fog"),
    ("brokkr", "noise"),
    ("imagenet-c", "fog"),
    ("imagenet-c", "gaussian_noise"),
}


class LabelError(ValueError):
    """The run's files cannot make a valid label (a record fails its check, a file changed, ...)."""


# ---- the numbers ----


def threshold(logits, labels) -> float:
    """The 90% prediction-set threshold from a build's clean conformal-calibration scores (LAC)."""
    return conformal_threshold(softmax(logits), labels, TARGET_COVERAGE)


def reliability(logits, labels, q: float) -> dict:
    """ECE, E-AURC and coverage with set size, each (value, 95% interval): scripts/31's calls, unchanged."""
    cal = calibration(logits, labels)
    sel = selective_prediction(*confidence_and_correct(logits, labels))
    sets = evaluate_sets(softmax(logits), labels, q)
    return {
        "ece": (cal["ece"], cal["ece_ci95"]),
        "e_aurc": (sel["e_aurc"], sel["e_aurc_ci95"]),
        "coverage": (sets["coverage"], sets["coverage_ci95"]),
        "mean_set_size": (sets["mean_set_size"], sets["mean_set_size_ci95"]),
    }


# ---- the sentences ----


def random_condition_sentence(path=SEED_SENSITIVITY) -> str:
    """H's approved sentence (note of 10 October 2026), every number read from the seed-sensitivity record:
    the largest change in points, with one decimal, rounded up; the model; the image count."""
    path = Path(path)
    if not path.is_file():
        raise LabelError(f"the random-condition limit needs Brokkr's record {path.name} (results/checks/)")
    record = json.loads(path.read_text(encoding="utf-8"))
    largest = max(abs(c["difference"]) for c in record["raw"]["cells"]) * 100
    figure = math.ceil(round(largest * 10, 6)) / 10  # rounded up, never down; round() drops float dust
    model = load_model_list()[record["model"]]["display_name"]
    return (
        f"Fog and noise damage use a random pattern. In one exploratory check ({model}, "
        f"{record['settings']['n_images']:,} images), changing only that pattern moved top-1 by up to "
        f"{figure:.1f} point{'' if figure == 1 else 's'}; on other models, or with fewer images, it may be "
        f"larger. The intervals on this label cover the choice of images only, not the choice of pattern."
    )


def warning_limit(w: dict) -> str:
    """The limits sentence each warning gives (docs/label_schema.md, note of 6 October 2026, 2)."""
    kind = w["kind"]
    if kind == "few images":
        part = w["part"].replace("_", "-")
        return (
            f"The {part} part has only {w['n_items']:,} images ({w['recommended']:,} or more are "
            f"recommended), so the numbers that rest on it are less precise."
        )
    if kind == "classes with few test images":
        return (
            f"Some classes have fewer than {w['below']} test images each "
            f"({', '.join(map(str, w['classes']))}); results for them rest on few images."
        )
    if kind == "low agreement with FP32":
        return (
            f"The INT8 build's first answer matched FP32's on only {w['agreement']:.2%} of {w['n_items']:,} "
            f"images (below {w['warn_below']:.0%}), so it behaves noticeably differently from FP32."
        )
    return (
        f"Only {w['usable']} reference predictions could be checked (fewer than {w['below']}), so a "
        f"preprocessing mistake is less likely to have been caught."
    )


def user_limits(plan: dict, hw: dict, warnings: list, conditions: list, seed_sensitivity) -> list:
    """The study's general sentences (with decision 6's machine sentence), then the user ones that apply."""
    crop = plan["preprocessing"]["crop"]
    imagenet_c = any(c["suite"] == "imagenet-c" for c in conditions)
    limits = general_limits(hw["cpu_model"], hw["os"], len(conditions) - 1, MACHINE_SENTENCE, imagenet_c)
    limits += [warning_limit(w) for w in warnings]
    if crop != STUDY_SIZE:  # D8
        limits.append(
            f"Damage was applied at this model's input size, {crop} x {crop} pixels; Brokkr's severities "
            f"were designed for {STUDY_SIZE} x {STUDY_SIZE} pictures, so these results are not "
            f"comparable with Brokkr study labels."
        )
    if not imagenet_c:  # H, 17 November 2026, option (b)
        limits.append(
            f"The 8 ImageNet-C conditions were not tested: their damage code needs pictures of at "
            f"least 32 x 32 pixels, and this model's input is {crop} x {crop}."
        )
    if any((c["suite"], c["corruption"]) in RANDOM_DAMAGE for c in conditions):
        limits.append(random_condition_sentence(seed_sensitivity))
    return limits + [
        "The licences, and the declaration that the images were not used to train the model, are the "
        "submitter's; Brokkr has not checked them.",
        "Near-duplicate images (the same picture resized or saved again) are not detected; only identical "
        "files are.",
        "Photo-rotation tags (EXIF orientation) are not applied: each image is read as it is stored.",
        "The check against the submitter's stated accuracy catches a large preprocessing or class-order "
        "mistake, but may pass a small one.",
    ]


# ---- the label ----


def make_label(plan: dict, out, seed_sensitivity=SEED_SENSITIVITY) -> dict:
    """The label of the run in `out` (run_plan.json, int8_build.json when Brokkr built INT8, records/)."""
    out = Path(out)
    used = {}  # path in --out -> (SHA-256, what checked it)

    def note(rel: str, check: str) -> None:
        if not (out / rel).is_file():
            raise LabelError(f"{rel} is missing from --out")
        used[rel] = (sha256_of(out / rel), check)

    def source(rel: str, field: str) -> dict:
        return {"file": rel, "sha256": used[rel][0], "field": field}

    note(PLAN_FILE, "written by the checks of brokkr-edge test")
    on_disk = json.loads((out / PLAN_FILE).read_text(encoding="utf-8"))
    if on_disk != plan:
        raise LabelError(f"{PLAN_FILE} in --out is not this run's plan")
    name, mid = plan["name"], plan["model_id"]
    fp32 = plan["models"]["fp32"]

    # Builds.
    reference = build_entry(
        build_id(mid, "fp32", fp32["sha256"]),
        "reference",
        "FP32 (full precision)",
        "fp32",
        {"method": "FP32 ONNX model, supplied by the submitter"},
        fp32["sha256"],
        fp32["size_bytes"],
        "usable",
        None,
    )
    if "shrunk" in plan["models"]:
        shrunk, agree = plan["models"]["shrunk"], plan["checks"]["agreement_with_fp32"]
        precision, status = INT8_SUPPLIED, shrunk["status"]
        failure = (
            None
            if status == "usable"
            else failure_block(
                "top-1 agreement with FP32",
                agree["agreement"],
                agree["failed_below"],
                agree["n_items"],
                "conformal_calibration",
                [source(PLAN_FILE, "checks.agreement_with_fp32")],
            )
        )
        labelled = build_entry(
            build_id(mid, "int8", shrunk["sha256"]),
            "labelled",
            "INT8 (supplied by the submitter)",
            "int8",
            {"method": SUPPLIED, "made_by": shrunk["made_by"], "found": shrunk["found"]},
            shrunk["sha256"],
            shrunk["size_bytes"],
            status,
            failure,
        )
    else:
        note(BUILD_FILE, "written by brokkr-edge test's INT8 build (brokkr_edge.user_build)")
        built = json.loads((out / BUILD_FILE).read_text(encoding="utf-8"))
        if built["fp32_sha256"] != fp32["sha256"]:
            raise LabelError(f"{BUILD_FILE} is the build of another FP32 model")
        agree = (built["checks"] or {}).get("agreement_with_fp32")
        precision, status, r = INT8_BROKKR, built["status"], built["recipe"]
        failure = (
            None
            if status == "usable"
            else failure_block(
                *(built["failure"][k] for k in ("check", "value", "limit", "n_items", "split")),
                [source(BUILD_FILE, "failure")],
            )
        )
        cal = built["calibration"]
        recipe = {
            "method": r["calibration"],
            "detail": r["description"],
            "format": r["format"],
            "weights": r["weights"],
            "activations": r["activations"],
            "calibration": {
                "dataset_id": f"user/{name}:int8_calibration",
                "n_items": cal["n_items"],
                "group_items": r["group_images"],
                "source": cal["source"],
            },
            "kept_float": None,
            "skip_symbolic_shape": r["skip_symbolic_shape"],
        }
        file = built["file"]
        labelled = build_entry(
            build_id(mid, "int8", file["sha256"]) if file else f"{mid}#int8:no-file",
            "labelled",
            INT8_METHODS[r["calibration"]]["display_name"],
            "int8",
            recipe,
            file["sha256"] if file else None,
            file["size_bytes"] if file else None,
            status,
            failure,
        )
        if not file:
            labelled["file"] = None  # decision 10: a build that crashed twice has no file
    builds = {"reference": reference, "labelled": labelled}
    status = {"reference": "usable", "labelled": labelled["status"]}
    precisions = {"reference": FP32, "labelled": precision}

    # Records: every usable build, the test part under each condition and the clean calibration part.
    conditions = plan["conditions"]
    records, arrays = {}, {}
    for role, prec in precisions.items():
        if status[role] != "usable":
            continue
        for part, conds in (("test", conditions), ("conformal_calibration", conditions[:1])):
            for cond in conds:
                path = record_path(out, prec, part, cond)
                rel = path.relative_to(out).as_posix()
                try:
                    record, scores = load_measurement(path)
                except (OSError, ValueError) as e:
                    raise LabelError(f"{rel}: {type(e).__name__}: missing, or its scores changed") from e
                problems = check_record(record)
                if problems or record["source"] != "user-submitted" or record["precision"] != prec:
                    raise LabelError(
                        f"{rel} fails its check: {problems or 'not this build or not a user record'}"
                    )
                note(rel, "brokkr_edge.schema.check_record")
                note(Path(rel).with_suffix(".npz").as_posix(), "checksum equal to its record's")
                key = (role, part, condition_id(cond["suite"], cond["corruption"], cond["severity"]))
                records[key], arrays[key] = (rel, record), scores

    # One machine, the same images in every condition.
    _, first = records[("reference", "test", "clean")]
    fp = first["device"]["fingerprint"]
    for rel, record in records.values():
        f = record["device"]["fingerprint"]
        if (f["cpu_model"], f["os"], f["os_release"]) != (fp["cpu_model"], fp["os"], fp["os_release"]):
            raise LabelError(f"{rel} was measured on another machine")
    for part in ("test", "conformal_calibration"):
        labels = {tuple(a["labels"].tolist()) for (_, p, _), a in arrays.items() if p == part}
        if len(labels) != 1:
            raise LabelError(f"the {part} records do not hold the same images")
    os_name = f"{fp['os']} {fp['os_release']}"
    hw_id = hardware_id(plan["device_kind"], fp["cpu_model"], os_name)
    hardware = [
        {
            "hardware_id": hw_id,
            "kind": plan["device_kind"],
            "cpu_model": fp["cpu_model"],
            "os": os_name,
            "architecture": fp.get("architecture"),
            "cores": {"types": fp.get("core_types"), "map": None},
            "features": (fp.get("cpu_features") or {}).get("features"),
            "fingerprint_source": source(records[("reference", "test", "clean")][0], "device.fingerprint"),
        }
    ]
    runtimes, runtime_of = [], {}
    for key, (rel, record) in records.items():
        entry = runtime_entry(record["runtime"], [source(rel, "runtime")])
        if entry["runtime_id"] not in {r["runtime_id"] for r in runtimes}:
            runtimes.append(entry)
        runtime_of[key] = entry["runtime_id"]

    # Datasets: the user's parts (no file name: each named by its folder fingerprint).
    parts = plan["split"]["parts"]
    test_id, cal_id = f"user/{name}:test", f"user/{name}:conformal_calibration"
    images = plan["licences"]
    datasets = []
    for part, others in (("test", [cal_id]), ("conformal_calibration", [test_id]), ("int8_calibration", [])):
        p = parts.get(part)
        if p is None:
            continue
        datasets.append(
            {
                "dataset_id": f"user/{name}:{part}",
                "name": f"the submitter's images ({name})",
                "split": part,
                "n_items": p["n_items"],
                "item_kind": "image",
                "licence": images["images"],
                "source": images.get("images_source"),
                "split_mode": plan["split"]["mode"],
                "folder_fingerprint": p["fingerprint"],
                "labelled": p["source"] != "calib-images",
                "disjoint_from": others,
            }
        )

    # Measurements: copied from the records, the reliability numbers worked out here, then the derived ones.
    cond_block = conditions_block(conditions)
    ids = [c["condition_id"] for c in cond_block]
    measurements, details, correct = [], {}, {}
    for role in precisions:
        if status[role] != "usable":
            continue
        bid = builds[role]["build_id"]
        cal_rel, cal_record = records[(role, "conformal_calibration", "clean")]
        cal_scores = arrays[(role, "conformal_calibration", "clean")]
        q = threshold(cal_scores["logits"], cal_scores["labels"])
        n_cal = cal_record["data"]["n_images"]
        details[bid] = {"target": TARGET_COVERAGE, "threshold": q, "calibration_items": n_cal}
        for cid in ids:
            rel, record = records[(role, "test", cid)]
            scores, n, rt = (
                arrays[(role, "test", cid)],
                record["data"]["n_images"],
                runtime_of[(role, "test", cid)],
            )
            npz = Path(rel).with_suffix(".npz").as_posix()
            correct[(role, cid)] = top1_correct(scores["logits"], scores["labels"])
            for metric in ("top1", "top5"):
                m = record["metrics"][metric]
                measurements.append(
                    measurement(
                        metric,
                        bid,
                        test_id,
                        cid,
                        hw_id,
                        rt,
                        m["value"],
                        m["ci95"],
                        n,
                        {},
                        [source(rel, f"metrics.{metric}")],
                    )
                )
            numbers = reliability(scores["logits"], scores["labels"], q)
            for metric, (value, ci) in numbers.items():
                settings, sources = {}, [source(npz, "logits, labels")]
                if metric in ("coverage", "mean_set_size"):
                    settings = {
                        "target_coverage": TARGET_COVERAGE,
                        "threshold": q,
                        "calibration_dataset_id": cal_id,
                        "calibration_items": n_cal,
                        "method": "LAC",
                    }
                    sources.append(source(Path(cal_rel).with_suffix(".npz").as_posix(), "logits, labels"))
                measurements.append(
                    measurement(
                        metric,
                        bid,
                        test_id,
                        cid,
                        hw_id,
                        rt,
                        float(value),
                        [float(x) for x in ci],
                        n,
                        settings,
                        sources,
                    )
                )
    score_sources = {
        (role, cid): source(
            Path(records[(role, "test", cid)][0]).with_suffix(".npz").as_posix(), "logits, labels"
        )
        for role in precisions
        if status[role] == "usable"
        for cid in ids
    }
    measurements += derived_measurements(
        builds,
        status,
        ids,
        correct,
        score_sources,
        {(role, cid): runtime_of[(role, "test", cid)] for role, cid in correct},
        test_id,
        hw_id,
    )
    rows = envelope_rows(builds, status, ids[1:], measurements, hw_id)

    # The user parts.
    warnings = [dict(w) for w in plan["warnings"]]
    if (
        precision == INT8_BROKKR
        and agree
        and status["labelled"] == "usable"
        and agree["agreement"] < WARN_AGREEMENT_BELOW
    ):
        warnings.append(
            {
                "kind": "low agreement with FP32",
                "agreement": agree["agreement"],
                "n_items": agree["n_items"],
                "warn_below": WARN_AGREEMENT_BELOW,
            }
        )
    expected = plan["checks"]["expected_accuracy"]
    checks = {
        "fp32_sanity": {
            "measured": expected["measured"],
            "ci95": expected["ci95"],
            "n_items": expected["n_items"],
            "dataset_id": cal_id,
            "published": {
                "value": expected["stated"],
                "source": "stated by the submitter",
                "n_images": expected["stated_n_images"],
                "measured_on": expected["measured_on"],
            },
            "tolerance": expected["tolerance"],
            "half_width_99": expected["half_width_99"],
            "allowed_gap": expected["allowed_gap"],
            "chance": expected["chance"],
            "pass": expected["pass"],
            "sources": [source(PLAN_FILE, "checks.expected_accuracy")],
        },
        "outputs": {**plan["checks"]["outputs"], "sources": [source(PLAN_FILE, "checks.outputs")]},
        "reference_predictions": plan["checks"]["reference_predictions"],
        "agreement_with_fp32": None
        if agree is None
        else {
            **agree,
            "sources": [
                source(BUILD_FILE if precision == INT8_BROKKR else PLAN_FILE, "checks.agreement_with_fp32")
            ],
        },
    }
    speed = [
        {
            "build_id": b["build_id"],
            "hardware_id": hw_id,
            "hardware_kind": plan["device_kind"],
            "runtime_id": None,
            "status": "not measured",
            "reason": USER_SPEED_REASON,
            "settings": {},
        }
        for b in (reference, labelled)
    ]
    licences = {
        "brokkr_code": "Apache-2.0",
        "model_code": images["model_code"],
        "model_weights": images["model_weights"],
        "images": images["images"],
        "declared_by_submitter": True,
        "label_data": LABEL_DATA,
        "sources": [source(PLAN_FILE, "licences")],
    }
    crop, prep = plan["preprocessing"]["crop"], plan["preprocessing"]
    model = {
        "model_id": mid,
        "name": name,
        "display_name": name,
        "publisher": "user",
        "weights": f"the submitter's FP32 file (SHA-256 {fp32['sha256'][:12]})",
        "task": "classification",
        "modality": "image",
        "input": {
            "name": fp32["input"]["name"],
            "shape": [3, crop, crop] if prep["layout"] == "NCHW" else [crop, crop, 3],
            "preprocessing": prep,
        },
        "outputs": {
            "n_classes": len(plan["classes"]),
            "class_set": f"user/{name}",
            "class_names": plan["classes"],
            "declared": plan["checks"]["outputs"]["declared"],
        },
        "licence": {"code": images["model_code"], "weights": images["model_weights"]},
    }
    git = git_info()
    generated = {**generated_block(git["commit"], git["dirty"], LABEL_DATA), "seed_scheme": USER_SEED_SCHEME}
    label = assemble(
        source={
            "kind": "user-submitted",
            "verified": False,
            "submitted_by": plan.get("submitted_by"),
            "how_made": plan["made_by"].replace("brokkr-edge", "brokkr-edge test", 1),
        },
        generated=generated,
        model=model,
        builds=builds,
        hardware=hardware,
        runtimes=runtimes,
        datasets=datasets,
        conditions=cond_block,
        checks=checks,
        measurements=measurements,
        rows=rows,
        speed=speed,
        details={"conformal": details},
        limits=user_limits(plan, hardware[0], warnings, conditions, seed_sensitivity),
        licences=licences,
        sources=[{"file": f, "sha256": sha, "check": check} for f, (sha, check) in sorted(used.items())],
    )
    label["declarations"] = dict(plan["declarations"])
    label["summary"]["warnings"] = warnings
    problems = check_label(label)
    if problems:
        raise LabelError(f"the label fails its check: {problems}")
    return label


def write_label(label: dict, out) -> list:
    """label.json, label.md and label.html in --out (LF line endings), if every number on the pages is a
    label.json number; returns their paths."""
    out = Path(out)
    md, page = to_markdown(label), to_html(label)
    loose = unexplained_numbers(label, md) + unexplained_numbers(label, page)
    if loose:
        raise LabelError(f"numbers on the label's pages that are not label.json numbers: {loose[:5]}")
    paths = []
    for name, text in (
        ("label.json", json.dumps(label, indent=2) + "\n"),
        ("label.md", md),
        ("label.html", page),
    ):
        path = out / name
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
        paths.append(path)
    return paths
