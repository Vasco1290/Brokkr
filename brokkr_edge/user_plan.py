"""Check a user's model and images before anything is run on them, and write the run plan (step 3, slice 1).

docs/user_models.md fixes the rules; this module runs the checks in order and stops at the first stage that
finds problems (raising UserInputError with every problem of that stage):

1. the settings file (brokkr_edge.user_settings);
2. the models, and a supplied pair (brokkr_edge.user_checks);
3. the image folders: listing, duplicates, unreadable images, the calibration folder's overlap
   (brokkr_edge.user_images);
4. the split into parts, the floors and the warnings (brokkr_edge.user_images.plan_split);
5. the outputs (logits or probabilities) and the shrunk build's agreement with FP32, on the first 256
   conformal-calibration images; the optional reference predictions;
6. the expected accuracy: clean FP32 top-1 on the conformal-calibration images against the user's own
   figure (never on the test images: the test part decides nothing; note of 7 October 2026).

While it runs: Python-level network connections blocked and counted; ONNX Runtime telemetry switched off
(brokkr_edge.no_network; the wording is H's, 7 October 2026).
The run plan names no image file and no absolute path: only counts, class names and fingerprints (the split
is remade from the folder by its fixed seeds, and the fingerprints show it is the same folder). Running the
damage conditions and making the label come in the next slice.
"""

import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr_edge import __version__
from brokkr_edge.benchmark import make_session
from brokkr_edge.no_network import no_network
from brokkr_edge.results import sha256_of, written_atomically
from brokkr_edge.user_checks import (
    BATCH,
    CHECK_THREADS,
    WARN_AGREEMENT_BELOW,
    agreement,
    check_model,
    check_outputs,
    check_pair,
    expected_accuracy,
    inspect_model,
    run,
)
from brokkr_edge.user_images import (
    CAPS,
    FLOORS,
    INT8_CALIBRATION_IMAGES,
    RECOMMENDED,
    SEEDS,
    fingerprint,
    overlap,
    plan_split,
    scan_labelled,
    scan_unlabelled,
)
from brokkr_edge.user_settings import UserInputError, load_settings

PLAN_VERSION = 1
AGREEMENT_IMAGES = 256  # the first 256 conformal-calibration images (never test images)
REPOSITORY = Path(__file__).resolve().parent.parent
PROTECTED = ("published/labels", "labels", "results")  # released labels, working study labels, study records
NETWORK_WORDING = "Python-level network connections blocked and counted; ONNX Runtime telemetry switched off"


def check_out_folder(out) -> list:
    """Problems with --out: not inside this repository's released labels, study labels or records."""
    if not (REPOSITORY / "ROADMAP.md").is_file():  # an installed package, not a Brokkr checkout
        return []
    out = Path(out).resolve()
    return [f"--out must not be inside {REPOSITORY / p}: user records stay apart from Brokkr's own"
            for p in PROTECTED if out == REPOSITORY / p or (REPOSITORY / p) in out.parents]


def _model_facts(path: Path, info: dict) -> dict:
    return {"sha256": sha256_of(path), "size_bytes": path.stat().st_size,
            "input": info["inputs"][0], "output": info["outputs"][0], "found": info["found"]}


def _stop_if(problems: list) -> None:
    if problems:
        raise UserInputError(problems)


def check_user_inputs(settings_path, images, calib_images=None, log=print) -> dict:
    """Run every check and return the run plan. Raises UserInputError (the inputs cannot be used)."""
    with no_network() as attempts:
        telemetry = "not available"
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
            telemetry = "switched off"
        plan = _check(Path(settings_path), Path(images), Path(calib_images) if calib_images else None, log)
    if attempts:  # an attempt some library caught and hid: still not allowed
        raise RuntimeError(f"network connections were attempted (and blocked): {attempts}")
    plan["network"] = {"what": NETWORK_WORDING, "connection_attempts": len(attempts),
                       "onnxruntime_telemetry_events": telemetry}
    return plan


def _check(settings_path: Path, images: Path, calib_images: Path | None, log) -> dict:
    settings, base = load_settings(settings_path)
    classes, prep, declared = settings["classes"], settings["preprocessing"], settings["outputs"]
    log(f"settings file: PASS ({settings['name']}, {len(classes)} classes)")

    fp32_path = base / settings["fp32_model"]
    shrunk = settings["shrunk_model"]
    shrunk_path = base / shrunk["file"] if shrunk else None
    infos, problems = {}, []
    for role, path in (("fp32", fp32_path), ("shrunk", shrunk_path)):
        if path is None:
            continue
        try:
            infos[role] = inspect_model(path)
        except ValueError as e:
            problems.append(f"{'the FP32 model' if role == 'fp32' else 'the shrunk build'}: {e}")
            continue
        problems += check_model(infos[role], prep, len(classes), role)
    _stop_if(problems)
    models = {role: _model_facts(path, infos[role])
              for role, path in (("fp32", fp32_path), ("shrunk", shrunk_path)) if path is not None}
    if shrunk:
        _stop_if(check_pair(infos["fp32"], infos["shrunk"], models["fp32"]["sha256"],
                            models["shrunk"]["sha256"]))
        models["shrunk"].update({"precision": shrunk["precision"], "made_by": shrunk["made_by"]})
    log(f"models: PASS ({'FP32 + supplied INT8 build' if shrunk else 'FP32; Brokkr will build INT8'})")

    labelled = scan_labelled(images, classes, settings["split"])
    entries = labelled["images"]
    unlabelled = scan_unlabelled(calib_images) if calib_images else None
    if unlabelled:
        _stop_if(overlap(entries, unlabelled["images"]))
    extra = f", {len(unlabelled['images'])} unlabelled" if unlabelled else ""
    log(f"images: PASS ({len(entries)} labelled{extra})")

    labels = [e["label"] for e in entries]
    split = plan_split(labels, [e["part"] for e in entries], len(classes), settings["split"],
                       len(unlabelled["images"]) if unlabelled else None, brokkr_builds_int8=shrunk is None)
    warnings = split["warnings"]
    for w in warnings:
        if w["kind"] == "classes with few test images":
            w["classes"] = [classes[c] for c in w["classes"]]
    log(f"split: PASS ({len(split['conformal_calibration'])} conformal calibration, "
        f"{len(split['test'])} test)")

    batch = 1 if isinstance(models["fp32"]["input"]["shape"][0], int) else BATCH
    sessions = {role: make_session(path, CHECK_THREADS)
                for role, path in (("fp32", fp32_path), ("shrunk", shrunk_path)) if path is not None}
    conformal = split["conformal_calibration"]
    conformal_out = run(sessions["fp32"], [images / entries[i]["rel"] for i in conformal], prep, batch)
    sample = conformal[:AGREEMENT_IMAGES]
    sample_paths = [images / entries[i]["rel"] for i in sample]
    fp32_out = conformal_out[:AGREEMENT_IMAGES]
    _stop_if(check_outputs(fp32_out, declared, len(classes), "fp32"))
    checks = {"outputs": {"declared": declared, "pass": True, "n_items": len(sample)}}
    if shrunk:
        shrunk_out = run(sessions["shrunk"], sample_paths, prep, batch)
        _stop_if(check_outputs(shrunk_out, declared, len(classes), "shrunk"))
        checks["agreement_with_fp32"] = agreement(fp32_out, shrunk_out)
        models["shrunk"]["status"] = checks["agreement_with_fp32"]["status"]
        value = checks["agreement_with_fp32"]["agreement"]
        if value < WARN_AGREEMENT_BELOW:
            warnings.append({"kind": "low agreement with FP32", "agreement": value, "n_items": len(sample),
                             "warn_below": WARN_AGREEMENT_BELOW})
        log(f"shrunk build: top-1 agreement with FP32 {value:.1%} on {len(sample)} images "
            f"({models['shrunk']['status']})")

    refs = settings.get("reference_predictions")
    checks["reference_predictions"] = None
    if refs:
        by_rel = {e["rel"]: e for e in entries}
        missing = [r["file"] for r in refs if r["file"] not in by_rel]
        _stop_if([f"reference_predictions: {f} is not an image of the labelled folder" for f in missing])
        top1 = run(sessions["fp32"], [images / r["file"] for r in refs], prep, batch).argmax(axis=1)
        wrong = [f"{r['file']}: Brokkr's FP32 says {classes[t]!r}, your code said {r['top1']!r}"
                 for r, t in zip(refs, top1, strict=True) if classes[t] != r["top1"]]
        _stop_if([f"reference prediction differs (check the preprocessing): {w}" for w in wrong])
        checks["reference_predictions"] = {"checked": len(refs), "matched": len(refs)}
        log(f"reference predictions: PASS ({len(refs)} of {len(refs)})")

    # On the conformal-calibration images, never the test images: this check can stop a run, and the test
    # part decides nothing (docs/user_models.md, note of 7 October 2026).
    correct = (conformal_out.argmax(axis=1) == np.asarray([labels[i] for i in conformal])).astype(np.int64)
    check = {"part": "conformal_calibration",
             **expected_accuracy(correct, settings["expected_accuracy"], len(classes))}
    checks["expected_accuracy"] = check
    if not check["pass"]:
        why = [] if check["within_tolerance"] else [
            f"clean FP32 top-1 {check['measured']:.1%} on {check['n_items']} conformal-calibration images, "
            f"against your stated {check['stated']:.1%} (allowed difference {check['tolerance']:.0%})"]
        if not check["above_chance"]:
            why.append(f"the lower end of its 95% interval ({check['ci95'][0]:.1%}) is not above chance "
                       f"({check['chance']:.1%})")
        raise UserInputError(why + ["check the class order, mean and std, channel order and resize"])
    log(f"expected accuracy: PASS ({check['measured']:.1%} measured, {check['stated']:.1%} stated)")

    def part(indices, source="labelled"):
        if source == "labelled":
            members = [entries[i] for i in indices]
            counts = [int(n) for n in np.bincount([labels[i] for i in indices], minlength=len(classes))]
        else:  # the unlabelled --calib-images folder: no classes to count
            members, counts = [unlabelled["images"][i] for i in indices], None
        return {"source": source, "n_items": len(indices), "per_class": counts,
                "fingerprint": fingerprint(members)}

    int8 = split["int8_calibration"]
    return {
        "kind": "brokkr-edge user run plan",
        "plan_version": PLAN_VERSION,
        "made_by": f"brokkr-edge {__version__}",
        "made_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "name": settings["name"],
        "model_id": f"user/{settings['name']}@{models['fp32']['sha256'][:12]}",
        "models": models,
        "int8_build": ("supplied by the submitter" if shrunk else
                       f"made by Brokkr: Percentile 99.99, {INT8_CALIBRATION_IMAGES} calibration images"),
        "classes": classes,
        "preprocessing": prep,
        "split": {
            "mode": settings["split"], "seeds": SEEDS, "floors": FLOORS, "recommended": RECOMMENDED,
            "caps": CAPS,
            "parts": {
                "int8_calibration": part(int8["indices"], int8["source"]) if int8 else None,
                "conformal_calibration": part(split["conformal_calibration"]),
                "test": part(split["test"]),
            },
            "unused": split["unused"],
            "labelled_folder": {"n_items": len(entries), "fingerprint": fingerprint(entries),
                                "other_files": labelled["other_files"]},
            "calibration_folder": None if not unlabelled else {
                "n_items": len(unlabelled["images"]), "fingerprint": fingerprint(unlabelled["images"]),
                "other_files": unlabelled["other_files"]},
        },
        "checks": checks,
        "warnings": warnings,
        "licences": settings["licences"],
        "declarations": settings["declarations"],
        "device_kind": settings["device_kind"],
        "submitted_by": settings.get("submitted_by"),
        "runtime": {"onnxruntime": ort.__version__, "threads": CHECK_THREADS, "batch": batch},
    }


def write_plan(plan: dict, out) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_plan.json"
    with written_atomically(path) as tmp:
        tmp.write_text(json.dumps(plan, indent=2), encoding="utf-8", newline="\n")
    return path
