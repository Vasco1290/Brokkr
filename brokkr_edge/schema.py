"""The unified result format (schema version 2, Stage 4 onward) and a checker for it.

One record = one measurement: one model at one precision, run with one runtime on one device, on one
set of images under one condition. Everything needed to trace a number back to how it was made is a
required field, so a record that can't be traced fails the check.

    {
      "schema_version": 2,
      "kind": "accuracy",                # or calibration, conformal, selective, speed, levels, diagnostic
      "source": "brokkr",                # or "community-submitted" (hard rule 7)
      "model": {"name": ..., "weights": ..., "licence": ...},
      "precision": "int8_percentile99.99",
      "runtime": {"name": "onnxruntime", "version": ..., "execution_provider": ..., "threads": 4},
      "device": {"label": "laptop", "fingerprint": {...}},       # fingerprint from brokkr_edge.fingerprint
      "code": {"commit": ..., "dirty": false},
      "data": {"dataset": ..., "split": "test", "n_images": ..., "licence": ...},   # null for speed
      "condition": {"corruption": "fog", "suite": "imagenet-c", "severity": 3},    # null for speed
      "metrics": {"top1": {"value": ..., "ci95": [low, high]}, ...},
      "settings": {...},                 # everything else about how it was run
      "arrays": {"file": ..., "sha256": ...} or null,            # saved per-image outputs
      "derived_from": [...],             # files this record was computed or converted from
    }

Brokkr's own corruptions and ImageNet-C's share some names (fog, defocus blur), so a condition always
says which suite it comes from, and condition_label() always prints it: "fog (Brokkr) s3".

Stage 1-3 files use schema version 1. from_v1() converts them in memory (the files are not changed),
so everything from Stage 4 on can read old and new results the same way.
"""

import json
import math
from pathlib import Path

import numpy as np

from brokkr_edge.results import sha256_of, written_atomically

SCHEMA_VERSION = 2
KINDS = ("accuracy", "calibration", "conformal", "selective", "speed", "levels", "diagnostic")
SOURCES = ("brokkr", "community-submitted")
SUITES = {"brokkr": "Brokkr", "imagenet-c": "ImageNet-C"}
DEVICE_LABELS = ("laptop", "raspberry-pi-5", "cloud-arm")
ARM_ARCHITECTURES = ("aarch64", "arm64")

# Metrics each kind must have, and which of them need a 95% interval (measurement rules).
REQUIRED_METRICS = {
    "accuracy": {"top1": True},
    "calibration": {"ece": True},
    # Coverage is never reported without its average set size (standing rule).
    "conformal": {"coverage": True, "mean_set_size": True},
    "selective": {"e_aurc": True},
    "speed": {"p50_ms": False, "p95_ms": False, "p99_ms": False},
    "levels": {},
    # A diagnostic explains a result (e.g. where an INT8 model drifts from FP32); it is never a result.
    # Its tables go in the optional "raw" field; "settings" must name the script that made it.
    "diagnostic": {},
}
MIN_WARMUP_RUNS, MIN_TIMED_RUNS = 20, 100

REQUIRED_KEYS = ("schema_version", "kind", "source", "model", "precision", "runtime", "device", "code",
                 "data", "condition", "metrics", "settings", "arrays", "derived_from")


def metric(value, ci95=None) -> dict:
    """One metric: its measured value and, where there is one, its 95% interval."""
    return {"value": value, "ci95": None if ci95 is None else [ci95[0], ci95[1]]}


def condition(corruption: str = "clean", suite: str | None = None, severity: int = 0) -> dict:
    return {"corruption": corruption, "suite": suite, "severity": severity}


def condition_label(cond: dict | None) -> str:
    """Human name of a condition, always naming its suite: "clean", "fog (ImageNet-C) s3"."""
    if cond is None:
        return "no images (speed)"
    if cond["corruption"] == "clean":
        return "clean"
    return f"{cond['corruption'].replace('_', ' ')} ({SUITES[cond['suite']]}) s{cond['severity']}"


def make_measurement(kind: str, model: dict, precision: str, runtime: dict, device_label: str,
                     fingerprint: dict, data: dict | None, cond: dict | None, metrics: dict,
                     settings: dict, arrays: dict | None = None, derived_from=(),
                     source: str = "brokkr") -> dict:
    """Build a schema-2 record. The code version is taken from the fingerprint's git entry."""
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "source": source,
        "model": model,
        "precision": precision,
        "runtime": runtime,
        "device": {"label": device_label, "fingerprint": fingerprint},
        "code": {"commit": fingerprint["git"]["commit"], "dirty": fingerprint["git"]["dirty"]},
        "data": data,
        "condition": cond,
        "metrics": metrics,
        "settings": settings,
        "arrays": arrays,
        "derived_from": list(derived_from),
    }


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_hex(x, length: int) -> bool:
    return isinstance(x, str) and len(x) == length and all(c in "0123456789abcdef" for c in x)


def check_record(r: dict) -> list:
    """Every problem with a schema-2 record, as plain sentences. An empty list means it passes."""
    missing = [k for k in REQUIRED_KEYS if k not in r]
    if missing:
        return [f"missing fields: {missing}"]
    problems = []
    if r["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version is {r['schema_version']}, expected {SCHEMA_VERSION}")
    if r["kind"] not in KINDS:
        problems.append(f"unknown kind {r['kind']!r}")
    if r["source"] not in SOURCES:
        problems.append(f"source must be one of {SOURCES} (hard rule 7)")

    model = r["model"]
    if not (isinstance(model, dict) and model.get("name")):
        problems.append("model has no name")
    elif not model.get("licence"):
        problems.append(f"model {model['name']!r} has no licence recorded (hard rule 8)")
    if not (isinstance(r["precision"], str) and r["precision"]):
        problems.append("precision is empty")

    rt = r["runtime"]
    if not (isinstance(rt, dict) and all(rt.get(k) for k in ("name", "version", "execution_provider"))):
        problems.append("runtime needs name, version and execution_provider")
    elif not (isinstance(rt.get("threads"), int) and rt["threads"] >= 1):
        problems.append("runtime threads must be a whole number >= 1")

    problems += _check_device(r["device"])

    code = r["code"]
    if not (isinstance(code, dict) and _is_hex(code.get("commit"), 40)
            and isinstance(code.get("dirty"), bool)):
        problems.append("code needs a 40-character git commit and a dirty flag")

    if r["kind"] == "speed":
        if r["data"] is not None or r["condition"] is not None:
            problems.append("speed records use random input: data and condition must be null")
    else:
        problems += _check_data(r["data"]) + _check_condition(r["condition"])

    problems += _check_metrics(r["kind"], r["metrics"])
    if r["kind"] == "diagnostic" and not (isinstance(r["settings"], dict) and r["settings"].get("script")):
        problems.append("a diagnostic record must name the script that made it (settings.script)")
    if r["kind"] == "speed":
        s = r["settings"]
        if not (s.get("warmup_runs", 0) >= MIN_WARMUP_RUNS and s.get("timed_runs", 0) >= MIN_TIMED_RUNS):
            problems.append(f"speed needs >= {MIN_WARMUP_RUNS} warm-up and >= {MIN_TIMED_RUNS} timed runs")

    arrays = r["arrays"]
    if arrays is not None and not (isinstance(arrays, dict) and arrays.get("file")
                                   and _is_hex(arrays.get("sha256"), 64)):
        problems.append("arrays needs a file name and a SHA-256 checksum")
    if not isinstance(r["derived_from"], list):
        problems.append("derived_from must be a list")
    return problems


def _check_device(device) -> list:
    if not (isinstance(device, dict) and isinstance(device.get("fingerprint"), dict)):
        return ["device needs a label and a fingerprint"]
    label, fp = device.get("label"), device["fingerprint"]
    board = fp.get("board_model") or ""
    if label not in DEVICE_LABELS:
        return [f"device label must be one of {DEVICE_LABELS}"]
    if not (fp.get("os") and fp.get("os_release")):
        return ["the fingerprint must record the OS (os and os_release)"]
    # A label must match the machine it claims (hard rule 2: cloud ARM is never called a Pi).
    if label == "raspberry-pi-5" and not board.startswith("Raspberry Pi 5"):
        return ["labelled raspberry-pi-5, but the fingerprint's board is not a Raspberry Pi 5"]
    if label == "laptop" and board:
        return [f"labelled laptop, but the fingerprint names a board ({board!r})"]
    if label == "cloud-arm" and (board or str(fp.get("architecture", "")).lower() not in ARM_ARCHITECTURES):
        return ["labelled cloud-arm, but the fingerprint is not an ARM machine without a board"]
    return []


def _check_data(data) -> list:
    if not isinstance(data, dict) or not all(data.get(k) for k in ("dataset", "split")):
        return ["data needs dataset and split"]
    problems = []
    if not (isinstance(data.get("n_images"), int) and data["n_images"] > 0):
        problems.append("data needs the number of images")
    if not data.get("licence"):
        problems.append(f"dataset {data['dataset']!r} has no licence recorded (hard rule 8)")
    return problems


def _check_condition(cond) -> list:
    if not (isinstance(cond, dict) and isinstance(cond.get("corruption"), str)):
        return ["condition needs a corruption name ('clean' for none)"]
    if cond["corruption"] == "clean":
        if cond.get("suite") is not None or cond.get("severity") != 0:
            return ["a clean condition has no suite and severity 0"]
        return []
    if cond.get("suite") not in SUITES:
        return [f"condition {cond['corruption']!r} must name its suite, one of {list(SUITES)}"]
    if cond.get("severity") not in (1, 2, 3, 4, 5):
        return ["a damaged condition has severity 1-5"]
    return []


def _check_metrics(kind: str, metrics) -> list:
    if not (isinstance(metrics, dict) and metrics):
        return ["metrics is empty"]
    problems = []
    for name, m in metrics.items():
        if not (isinstance(m, dict) and _is_number(m.get("value"))):
            problems.append(f"metric {name!r} needs a finite number as its value")
            continue
        ci = m.get("ci95")
        if ci is not None and not (isinstance(ci, list) and len(ci) == 2 and all(map(_is_number, ci))
                                   and ci[0] <= ci[1]):
            problems.append(f"metric {name!r} has a malformed 95% interval")
    for name, needs_ci in REQUIRED_METRICS.get(kind, {}).items():
        if name not in metrics:
            problems.append(f"a {kind} record needs the metric {name!r}")
        elif needs_ci and metrics[name].get("ci95") is None:
            problems.append(f"metric {name!r} needs its 95% interval")
    return problems


def save_measurement(record: dict, path) -> Path:
    """Write a schema-2 record, refusing one that fails the check."""
    problems = check_record(record)
    if problems:
        raise ValueError("result record fails the schema check: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with written_atomically(path) as tmp:
        tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def load_measurement(path) -> tuple:
    """A schema-2 record and its saved per-image arrays, after checking the arrays' checksum."""
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    info = record["arrays"]
    npz_path = path.with_name(info["file"])
    if sha256_of(npz_path) != info["sha256"]:
        raise ValueError(f"{npz_path} does not match the checksum in {path}")
    with np.load(npz_path) as data:
        return record, {name: data[name] for name in data.files}


# ---- Model build records (models/*.json) ----

# Build options that change how a model is made, with their default. A record made by code that had an
# option must state it explicitly, so a non-default build can never hide (see check_build_record).
BUILD_OPTIONS = {"skip_symbolic_shape": False, "kept_float": None}  # kept_float: SE1 (brokkr_edge.se_float)
EXPORT_MAX_ABS_DIFF = 1e-4       # scripts/01_export_model.py
# Below this an FP16/INT8 build is broken (brokkr_edge.quantize.check_int8_build).
MIN_AGREEMENT_WITH_FP32 = 0.20


def build_sanity(r: dict) -> bool | None:
    """Did the build's own sanity check pass? None if the record holds no sanity check."""
    if "pytorch_vs_onnx" in r:
        check = r["pytorch_vs_onnx"]
        return check["max_abs_diff"] < EXPORT_MAX_ABS_DIFF and check["top1_agreement"] == 1.0
    if "build" in r:
        return all(r["build"]["checks"].values())
    check = r.get("sanity_check")
    if not check:
        return None
    if "checks" in check:
        return all(check["checks"].values())
    return check["top1_agreement_with_fp32"] >= MIN_AGREEMENT_WITH_FP32


def non_default_settings(r: dict) -> dict:
    """The build settings in this record that differ from Brokkr's defaults (shown by the checker)."""
    s = r.get("settings") or {}
    found = {k: s[k] for k, default in BUILD_OPTIONS.items() if k in s and s[k] != default}
    if s.get("unrounded_output_ops"):
        found["unrounded_output_ops"] = s["unrounded_output_ops"]
    calibration = s.get("calibration") or {}
    if calibration.get("group_images") not in (None, 128):
        found["calibration group_images"] = calibration["group_images"]
    if calibration.get("split") not in (None, "int8_calibration"):
        found["calibration images"] = f"{calibration['split']}, seed {calibration.get('seed')}"
    if calibration.get("held_out_corruption"):
        found["damaged calibration, held out"] = calibration["held_out_corruption"]
    if calibration.get("purpose"):
        found["purpose"] = calibration["purpose"]
    return found


def check_build_record(r: dict, options_that_existed: set) -> tuple:
    """(status, problems) for one model build record.

    status: "usable" (its sanity check passed), "failed" (it failed), or "no sanity check".
    problems: why the record itself is not acceptable (not clean, no licence, an option not stated).
    options_that_existed: the BUILD_OPTIONS the building code already had, so the record must state them.
    """
    problems = []
    git = (r.get("machine") or {}).get("git") or {}
    if not _is_hex(git.get("commit"), 40):
        problems.append("no git commit recorded")
    if git.get("dirty") is not False:
        problems.append("not built from a clean commit (dirty flag is not false)")
    if not r.get("licence"):
        problems.append("no licence recorded (hard rule 8)")
    if (r.get("settings") or {}).get("calibration_method"):  # a quantized build
        for option in options_that_existed:
            if option not in r["settings"]:
                problems.append(f"built by code that had the option {option!r}, "
                                "but the record does not state it")
    passed = build_sanity(r)
    status = "no sanity check" if passed is None else ("usable" if passed else "failed")
    return status, problems


# ---- Reading Stage 1-3 (schema version 1) records ----

V1_MEASUREMENT_KINDS = ("accuracy", "calibration", "conformal", "selective", "speed")
# Settings that move into their own schema-2 field instead of staying in "settings".
_MOVED_SETTINGS = ("dataset", "split", "n_images", "dataset_licence", "corruption", "severity",
                   "num_threads", "execution_provider", "onnxruntime_version")


def from_v1(r: dict, file: str, models: dict, datasets: dict, v1_by_file: dict,
            device_label: str = "laptop") -> dict | None:
    """Convert one schema-1 measurement record to schema 2, or None if it isn't a measurement.

    models / datasets: name -> {"weights"?, "licence"} (from brokkr_edge.export and brokkr_edge.datasets).
    v1_by_file: every schema-1 record by file name, used to find the model run that a reliability
    record was computed from (its runtime is that run's runtime).
    """
    if r.get("schema_version") != 1 or r.get("kind") not in V1_MEASUREMENT_KINDS:
        return None
    s = r["settings"]
    run = r if r["kind"] in ("accuracy", "speed") else v1_by_file.get(s.get("source_result"))
    if run is None:
        raise ValueError(f"{file}: its source result {s.get('source_result')!r} was not found")
    rs = run["settings"]

    spec = models.get(r["model"], {})
    model = {"name": r["model"], "weights": spec.get("weights"), "licence": spec.get("licence")}
    runtime = {
        "name": "onnxruntime",
        "version": rs.get("onnxruntime_version") or run["machine"]["packages"]["onnxruntime"],
        # Every Stage 1-3 session was made by brokkr_edge.benchmark.make_session, which uses the CPU provider.
        "execution_provider": rs.get("execution_provider", "CPUExecutionProvider"),
        "threads": rs["num_threads"],
    }
    data = cond = None
    if r["kind"] != "speed":
        dataset = s["dataset"]
        data = {"dataset": dataset, "split": s["split"], "n_images": s["n_images"],
                "licence": s.get("dataset_licence") or datasets.get(dataset, {}).get("licence")}
        corruption = s.get("corruption", "clean")
        cond = condition(corruption, None if corruption == "clean" else "brokkr", s.get("severity", 0))

    metrics, details = {}, {}
    settings = {k: v for k, v in s.items() if k not in _MOVED_SETTINGS}
    for name, value in r["metrics"].items():
        if name.endswith("_ci95"):
            continue
        if name == "n_bins":
            settings["n_bins"] = value
        elif _is_number(value):
            metrics[name] = metric(value, r["metrics"].get(name + "_ci95"))
        else:
            details[name] = value
    if details:
        settings["details"] = details
    if "reference" in r:
        settings["reference"] = r["reference"]

    derived = [{"file": file, "note": "converted from schema version 1; the file is unchanged"}]
    if run is not r:
        derived.append({"file": s["source_result"], "arrays_sha256": s.get("source_arrays_sha256")})
    arrays = None
    if "raw_arrays" in r:
        arrays = {"file": r["raw_arrays"]["file"], "sha256": r["raw_arrays"]["sha256"]}
    record = make_measurement(r["kind"], model, r["precision"], runtime, device_label, r["machine"], data,
                              cond, metrics, settings, arrays, derived, r["source"])
    if "raw" in r:
        record["raw"] = r["raw"]
    return record


def load_measurements(folder, models: dict, datasets: dict) -> tuple:
    """Every measurement under `folder` as schema 2, plus the files that are not measurements.

    Returns (measurements, skipped): measurements is a list of (path, record); skipped is a list of
    (path, kind) for decision records (choices, checks, verdicts), which have no single condition.
    """
    paths = sorted(Path(folder).rglob("*.json"))
    raw = {p: json.loads(p.read_text()) for p in paths}
    v1_by_file = {p.name: r for p, r in raw.items() if r.get("schema_version") == 1}
    if len(v1_by_file) != sum(r.get("schema_version") == 1 for r in raw.values()):
        raise ValueError("two schema-1 files share a name, so source results would be ambiguous")
    measurements, skipped = [], []
    for p, r in raw.items():
        if r.get("schema_version") == SCHEMA_VERSION:
            measurements.append((p, r))
            continue
        converted = from_v1(r, p.name, models, datasets, v1_by_file)
        if converted is None:
            skipped.append((p, r.get("kind")))
        else:
            measurements.append((p, converted))
    return measurements, skipped
