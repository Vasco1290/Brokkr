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
is remade from the folder by its fixed seeds, and the fingerprints show it is the same folder: see
rebuild_parts). The INT8 build, the conditions and the label follow (brokkr_edge.user_command).
"""

import datetime
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr_edge import __version__
from brokkr_edge.benchmark import make_session
from brokkr_edge.no_network import no_network
from brokkr_edge.results import sha256_of, written_atomically
from brokkr_edge.test_run import CONDITIONS
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
REFERENCES_TO_STOP = 20  # fewer usable reference predictions than this only warn (note of 7 October 2026)
REPOSITORY = Path(__file__).resolve().parent.parent
PROTECTED = ("published/labels", "labels", "results")  # released labels, working study labels, study records
NETWORK_WORDING = "Python-level network connections blocked and counted; ONNX Runtime telemetry switched off"
IMAGENET_C_SMALLEST = 32  # the ImageNet-C code refuses smaller pictures (tests/test_user_damage_sizes.py)


def conditions_for(crop: int) -> list:
    """The conditions a user run applies: the study's 13; below 32 x 32, clean and Brokkr's four, and the 8
    ImageNet-C conditions are not tested (H, 17 November 2026, option b; docs/user_models.md)."""
    if crop >= IMAGENET_C_SMALLEST:
        return [dict(c) for c in CONDITIONS]
    return [dict(c) for c in CONDITIONS if c["suite"] in (None, "brokkr")]


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
        # Only non-test images are used: an entry naming a test image is ignored and counted, never run, so a
        # test image can never stop the run (docs/user_models.md, second note of 7 October 2026).
        by_rel = {e["rel"]: i for i, e in enumerate(entries)}
        missing = [r["file"] for r in refs if r["file"] not in by_rel]
        _stop_if([f"reference_predictions: {f} is not an image of the labelled folder" for f in missing])
        in_test = set(split["test"])
        usable = [r for r in refs if by_rel[r["file"]] not in in_test]
        wrong = []
        if usable:
            top1 = run(sessions["fp32"], [images / r["file"] for r in usable], prep, batch).argmax(axis=1)
            wrong = [f"{r['file']}: Brokkr's FP32 says {classes[t]!r}, your code said {r['top1']!r}"
                     for r, t in zip(usable, top1, strict=True) if classes[t] != r["top1"]]
        can_stop = len(usable) >= REFERENCES_TO_STOP
        if can_stop:
            _stop_if([f"reference prediction differs (check the preprocessing): {w}" for w in wrong])
        else:  # too few to stop the run: a warning, with counts only (no file names in the run plan)
            warnings.append({"kind": "few usable reference predictions", "usable": len(usable),
                             "ignored_test_images": len(refs) - len(usable), "mismatched": len(wrong),
                             "below": REFERENCES_TO_STOP})
        checks["reference_predictions"] = {
            "given": len(refs), "ignored_test_images": len(refs) - len(usable), "checked": len(usable),
            "matched": len(usable) - len(wrong), "mismatched": len(wrong), "can_stop_the_run": can_stop}
        log(f"reference predictions: {len(usable) - len(wrong)} of {len(usable)} usable match; "
            f"{len(refs) - len(usable)} naming test images ignored"
            + ("" if can_stop else f" (fewer than {REFERENCES_TO_STOP} usable: a warning, not a stop)"))

    # On the conformal-calibration images, never the test images: this check can stop a run, and the test
    # part decides nothing (docs/user_models.md, note of 7 October 2026).
    correct = (conformal_out.argmax(axis=1) == np.asarray([labels[i] for i in conformal])).astype(np.int64)
    check = {"part": "conformal_calibration",
             **expected_accuracy(correct, settings["expected_accuracy"], len(classes))}
    checks["expected_accuracy"] = check
    if not check["pass"]:
        why = [] if check["within_tolerance"] else [
            f"clean FP32 top-1 {check['measured']:.1%} on {check['n_items']} conformal-calibration images, "
            f"against your stated {check['stated']:.1%} (allowed difference {check['allowed_gap']:.1%}: the "
            f"larger of {check['tolerance']:.0%} and the 99% half-width for {check['n_items']} images)"]
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
        "conditions": conditions_for(prep["crop"]),
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


def rebuild_parts(plan: dict, images, calib_images=None) -> dict:
    """Each part's images, remade from the folders by the plan's fixed seeds (the run plan names no file).

    Returns {part: [{"path", "sha256", "label"}, ...] in canonical order}; the INT8 calibration part is None
    when the user supplied the shrunk build.
    Stops (UserInputError) if a folder or a part differs from what the plan's fingerprints say: the folders
    changed after the checks ran, so the plan no longer describes them.
    """
    images, split = Path(images), plan["split"]
    entries = scan_labelled(images, plan["classes"], split["mode"])["images"]
    problems = []
    if fingerprint(entries) != split["labelled_folder"]["fingerprint"]:
        problems.append("the images folder changed after the checks ran; run the checks again")
    unlabelled = None
    if split["calibration_folder"] and calib_images is None:
        problems.append("the checks used --calib-images; give the same folder again")
    elif calib_images is not None and not split["calibration_folder"]:
        problems.append("the checks ran without --calib-images; leave it out")
    elif calib_images is not None:
        unlabelled = scan_unlabelled(calib_images)["images"]
        if fingerprint(unlabelled) != split["calibration_folder"]["fingerprint"]:
            problems.append("the --calib-images folder changed after the checks ran; run the checks again")
    _stop_if(problems)

    labels = [e["label"] for e in entries]
    parts = plan_split(labels, [e["part"] for e in entries], len(plan["classes"]), split["mode"],
                       len(unlabelled) if unlabelled is not None else None,
                       brokkr_builds_int8="shrunk" not in plan["models"])
    out = {}
    for name in ("int8_calibration", "conformal_calibration", "test"):
        if name == "int8_calibration":
            int8 = parts[name]
            if int8 is None:
                out[name] = None
                continue
            from_calib = int8["source"] == "calib-images"
            root, pool, indices = (Path(calib_images), unlabelled, int8["indices"]) if from_calib else (
                images, entries, int8["indices"])
        else:
            root, pool, indices = images, entries, parts[name]
        members = [pool[i] for i in indices]
        if fingerprint(members) != split["parts"][name]["fingerprint"]:
            problems.append(f"the {name} part differs from the plan's; run the checks again")
        out[name] = [{"path": root / e["rel"], "sha256": e["sha256"], "label": e["label"]} for e in members]
    _stop_if(problems)
    return out


VOLATILE = ("made_utc", "network")  # differ between two runs of the same inputs


def plan_fingerprint(plan: dict) -> str:
    """SHA-256 of the plan without its time and network counts: equal for two runs of the same inputs."""
    stable = {k: v for k, v in plan.items() if k not in VOLATILE}
    return sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()


def same_run_problems(plan: dict, out) -> list:
    """Decision 13: --out may hold this run (it is carried on), never a different one."""
    path = Path(out) / "run_plan.json"
    if not path.is_file():
        return []
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        old = {}
    if plan_fingerprint(old) != plan_fingerprint(plan):
        return [f"--out already holds a different run (its run_plan.json differs from these inputs'): use an "
                f"empty folder, or the same inputs to carry that run on ({path})"]
    return []


def write_plan(plan: dict, out) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_plan.json"
    with written_atomically(path) as tmp:
        tmp.write_text(json.dumps(plan, indent=2), encoding="utf-8", newline="\n")
    return path
