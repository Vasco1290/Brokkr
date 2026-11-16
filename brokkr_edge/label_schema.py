"""The label format (docs/label_schema.md, schema version 1): stable IDs and a validator.

check_label(label) returns every problem as a plain sentence; an empty list means the label follows
schema version 1. It checks structure and consistency only; whether each number equals its source
file is checked by scripts/40_check_labels.py.

A label of a user's own model (source.kind "user-submitted", step 3) also follows the rules of the note of
6 October 2026 in docs/label_schema.md and H's decisions of 10 October 2026 (docs/user_models.md); see
_check_user_label. study_label_problems() is the guard that keeps any user run's records out of the released
study labels (decision 1).
"""

import math
import re

SCHEMA_VERSION = 1
SOURCE_KINDS = ("official", "user-submitted")
MODALITIES = ("image", "signal")
DEVICE_KINDS = ("laptop", "raspberry-pi-5", "cloud-arm")
USER_DEVICE_KINDS = ("laptop", "desktop", "server", "raspberry-pi-5", "cloud-arm", "other")  # H, D11
BUILD_ROLES = ("reference", "labelled")
BUILD_STATUSES = ("usable", "failed")
STATES = ("not harmful", "harmful", "borderline", "not tested", "INT8 build failed")
# A row lists each shrinking-cost flag that applies, in this order (note of 3 October 2026).
SHRINKING_COST_FLAGS = ("large shrinking cost", "not informative")
# Summary: lines by the labelled build's own state, harmful ones also by cause (second note of 30
# September 2026 in docs/label_schema.md; the rule is brokkr_edge.label).
HARM_CAUSES = ("too hard for this model", "hurt by shrinking", "cause unclear")
FAILED_LINES = ("damage drop", "coverage")  # the lines a harmful row can fail, in the order shown
LINE_STATES = ("fails", "copes", "straddles")  # a row's state on one line (note of 3 October 2026)
SPEED_STATUSES = ("measured", "not measured")
# Metric name -> unit. Derived comparisons (damage_drop, shrinking_cost) are measurements too.
METRICS = {
    "top1": "fraction",
    "top5": "fraction",
    "coverage": "fraction",
    "mean_set_size": "classes",
    "ece": "fraction",
    "e_aurc": "fraction",
    "damage_drop": "fraction",
    "shrinking_cost": "fraction",
}
REQUIRED_KEYS = (
    "schema_version",
    "label_id",
    "source",
    "generated",
    "model",
    "builds",
    "hardware",
    "runtimes",
    "datasets",
    "conditions",
    "checks",
    "measurements",
    "envelope",
    "summary",
    "speed",
    "details",
    "limits",
    "licences",
    "sources",
)
LICENCE_FIELDS = ("brokkr_code", "model_code", "model_weights", "label_data")  # each must be non-empty
# What a measured laptop speed row copies from its latency record (note of 3 October 2026).
SPEED_FIELDS = ("p50_ms", "p95_ms", "p99_ms", "spread_pct", "unstable_above_pct")
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:/@#-]*$")
ABSOLUTE_PATH = re.compile(r"^([A-Za-z]:)?[/\\]")  # a path from this machine: "C:/...", "/home/..."
# A path from the user's machine anywhere inside a text (a user label holds no file names or paths).
MACHINE_PATH = re.compile(r"(^|[\s'\"(=])([A-Za-z]:[/\\]|/home/|/Users/|\\\\)")

# User labels (note of 6 October 2026 in docs/label_schema.md; H's decisions of 10 October 2026).
USER_KIND = "user-submitted"
USER_MODEL_ID = re.compile(r"^user/([a-z0-9][a-z0-9_.-]*)@([0-9a-f]{12})$")
USER_SEED_SCHEME = "content-v1"
USER_SPEED_REASON = "this version of brokkr-edge test does not measure speed"  # decision 8
SUPPLIED = "supplied by the submitter"
SPLIT_MODES = ("brokkr", "own")
Z99 = 2.576  # brokkr_edge.user_checks: the expected-accuracy check's 99% half-width
# Each warning kind and the fields it must hold (decision 9 added the fourth).
WARNING_FIELDS = {
    "few images": ("part", "n_items", "recommended"),
    "classes with few test images": ("classes", "below"),
    "low agreement with FP32": ("agreement", "n_items", "warn_below"),
    "few usable reference predictions": ("usable", "ignored_test_images", "mismatched", "below"),
}


def slug(text: str) -> str:
    """Lowercase, with every run of other characters turned into one hyphen: "Windows 11" -> "windows-11"."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def model_id(publisher: str, name: str, weights: str) -> str:
    """e.g. torchvision/mobilenet_v3_large@imagenet1k_v2 (the weights' own name, after the class name)."""
    return f"{publisher}/{name}@{weights.split('.')[-1]}".lower()


def build_id(model: str, precision: str, sha256: str) -> str:
    """The model ID, the precision and the first 12 hex characters of the build file's SHA-256."""
    return f"{model}#{precision}:{sha256[:12]}".lower()


def hardware_id(kind: str, cpu_model: str, os_name: str) -> str:
    return f"{kind}:{slug(cpu_model)}:{slug(os_name)}"


def dataset_id(publisher: str, dataset: str, split: str) -> str:
    return f"{slug(publisher)}/{slug(dataset)}:{slug(split)}"


def condition_id(suite: str | None, damage: str, severity: int | None) -> str:
    return "clean" if damage == "clean" else f"{slug(suite)}/{slug(damage)}/{severity}"


def runtime_id(name: str, version: str, provider: str, threads: int, spinning: str | None) -> str:
    """e.g. onnxruntime@1.23.2:cpuexecutionprovider:4t:spin-on. Spinning as the record states it ("on ...",
    "off"); a record that does not state it gets "spin-not-recorded", never a guess."""
    spin = "not-recorded" if not spinning else ("on" if str(spinning).startswith("on") else "off")
    return f"{slug(name)}@{version}:{slug(provider)}:{int(threads)}t:spin-{spin}"


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_sources(sources, where: str) -> list:
    if not (isinstance(sources, list) and sources):
        return [f"{where}: no sources"]
    return [
        f"{where}: a source needs a file and a 64-character SHA-256"
        for s in sources
        if not (s.get("file") and len(str(s.get("sha256", ""))) == 64)
    ] + [
        f"{where}: source {s['file']!r} must be a path from the repository folder, not from this machine"
        for s in sources
        if ABSOLUTE_PATH.match(str(s.get("file", "")))
    ]


def check_label(label: dict) -> list:
    """Every way `label` breaks schema version 1, as sentences. Empty = valid."""
    missing = [k for k in REQUIRED_KEYS if k not in label]
    if missing:
        return [f"missing fields: {missing}"]
    problems = []
    if label["schema_version"] != SCHEMA_VERSION:
        problems.append(
            f"schema_version {label['schema_version']!r}: this reader knows only {SCHEMA_VERSION}"
        )

    source = label["source"]
    if source.get("kind") not in SOURCE_KINDS:
        problems.append(f"source.kind must be one of {SOURCE_KINDS}")
    elif source.get("verified") is not (source["kind"] == "official"):
        problems.append("only an official label is verified; a user-submitted one is always unverified")
    generated = label["generated"]
    if generated.get("dirty") is not False and source.get("kind") == "official":
        problems.append("an official label must be made from a clean commit (generated.dirty = false)")

    model = label["model"]
    if not ID_PATTERN.match(str(model.get("model_id", ""))):
        problems.append("model.model_id is not a valid ID")
    if model.get("modality") not in MODALITIES:
        problems.append(f"model.modality must be one of {MODALITIES}")
    licence = model.get("licence") or {}
    if not (licence.get("code") and licence.get("weights")):
        problems.append("the model has no licence recorded (hard rule 8)")

    builds = {b.get("build_id"): b for b in label["builds"]}
    roles = sorted(b.get("role") for b in label["builds"])
    if roles != sorted(BUILD_ROLES):
        problems.append("builds must be exactly one reference and one labelled build")
    for b in label["builds"]:
        if not ID_PATTERN.match(str(b.get("build_id", ""))):
            problems.append(f"build ID {b.get('build_id')!r} is not valid")
        if b.get("status") not in BUILD_STATUSES:
            problems.append(f"build {b.get('build_id')}: status must be one of {BUILD_STATUSES}")
        if (b.get("status") == "failed") != bool(b.get("failure")):
            problems.append(f"build {b.get('build_id')}: a failed build needs its failure, a usable one none")
        if not (
            len(str((b.get("file") or {}).get("sha256", ""))) == 64
            and _finite((b.get("file") or {}).get("size_bytes"))
        ):
            problems.append(f"build {b.get('build_id')}: file needs a SHA-256 and a size")
    labelled = next((b for b in label["builds"] if b.get("role") == "labelled"), {})
    if label["label_id"] != labelled.get("build_id"):
        problems.append("label_id must equal the labelled build's build_id")

    hardware = {h.get("hardware_id"): h for h in label["hardware"]}
    kinds = USER_DEVICE_KINDS if source.get("kind") == USER_KIND else DEVICE_KINDS
    for h in label["hardware"]:
        if h.get("kind") not in kinds or not ID_PATTERN.match(str(h.get("hardware_id", ""))):
            problems.append(
                f"hardware {h.get('hardware_id')!r}: needs a valid ID and a kind in {kinds}"
            )
        if "runtime" in h:
            problems.append(f"hardware {h.get('hardware_id')!r}: runtime belongs in 'runtimes', not here")
    runtimes = {r.get("runtime_id"): r for r in label["runtimes"]}
    for r in label["runtimes"]:
        if not ID_PATTERN.match(str(r.get("runtime_id", ""))) or not (r.get("name") and r.get("version")):
            problems.append(f"runtime {r.get('runtime_id')!r}: needs a valid ID, a name and a version")
        problems += _check_sources(r.get("sources"), f"runtime {r.get('runtime_id')}")
    licences = label["licences"]
    if not all(licences.get(k) for k in LICENCE_FIELDS):
        problems.append(f"licences: each of {LICENCE_FIELDS} must be stated (hard rule 8)")
    else:
        problems += _check_sources(licences.get("sources"), "licences")
    datasets = {d.get("dataset_id"): d for d in label["datasets"]}
    for d in label["datasets"]:
        if not d.get("licence"):
            problems.append(f"dataset {d.get('dataset_id')}: no licence recorded (hard rule 8)")
    conditions = {c.get("condition_id"): c for c in label["conditions"]}
    for c in label["conditions"]:
        if c.get("modality") != model.get("modality"):
            problems.append(f"condition {c.get('condition_id')}: its modality differs from the model's")

    seen = set()
    for i, m in enumerate(label["measurements"]):
        where = f"measurement {i} ({m.get('metric')}, {m.get('build_id')}, {m.get('condition_id')})"
        if m.get("metric") not in METRICS:
            problems.append(f"{where}: unknown metric")
        if (
            m.get("build_id") not in builds
            or m.get("dataset_id") not in datasets
            or m.get("condition_id") not in conditions
            or m.get("hardware_id") not in hardware
            or m.get("runtime_id") not in runtimes
        ):
            problems.append(
                f"{where}: refers to a build, dataset, condition, hardware or runtime not in the label"
            )
        elif builds[m["build_id"]].get("status") == "failed":
            problems.append(f"{where}: a failed build has no measurements")
        ci = m.get("ci95")
        if not _finite(m.get("value")) or (
            ci is not None and not (len(ci) == 2 and all(map(_finite, ci)) and ci[0] <= ci[1])
        ):
            problems.append(f"{where}: value must be finite and the interval ordered")
        problems += _check_sources(m.get("sources"), where)
        seen.add(
            (
                m.get("metric"),
                m.get("build_id"),
                m.get("dataset_id"),
                m.get("condition_id"),
                m.get("hardware_id"),
            )
        )
    for metric, *rest in seen:
        if metric == "coverage" and ("mean_set_size", *rest) not in seen:
            problems.append(f"coverage without its average set size for {rest} (standing rule)")

    for row in label["envelope"].get("rows", []):
        if row.get("state") not in STATES:
            problems.append(f"envelope row {row.get('condition_id')}: unknown state {row.get('state')!r}")
        flags = row.get("shrinking_cost_flags")
        if not isinstance(flags, list) or flags != [f for f in SHRINKING_COST_FLAGS if f in flags]:
            problems.append(
                f"envelope row {row.get('condition_id')}: shrinking_cost_flags must be a list from "
                f"{SHRINKING_COST_FLAGS}, in that order"
            )
        if row.get("build_id") not in builds or row.get("condition_id") not in conditions:
            problems.append(
                f"envelope row {row.get('condition_id')}: refers to an unknown build or condition"
            )
        elif builds[row["build_id"]].get("status") == "failed" and row.get("state") != "INT8 build failed":
            problems.append(
                f"envelope row {row.get('condition_id')}: a failed build's rows say 'INT8 build failed'"
            )
        failed = row.get("failed", [])
        lines = row.get("line_states")
        if row.get("state") == "INT8 build failed":
            if lines is not None:
                problems.append(f"envelope row {row.get('condition_id')}: a failed build has no line states")
        elif (
            not isinstance(lines, dict)
            or sorted(lines) != sorted(FAILED_LINES)
            or any(v not in LINE_STATES for v in lines.values())
            or [x for x in FAILED_LINES if lines[x] == "fails"] != failed
            or (row.get("state") == "not harmful") != all(v == "copes" for v in lines.values())
        ):
            problems.append(
                f"envelope row {row.get('condition_id')}: line_states must give {LINE_STATES} for each of "
                f"{FAILED_LINES}, agreeing with its state and failed lines"
            )
        in_order = [x for x in FAILED_LINES if x in failed] == failed
        if not in_order or bool(failed) != (row.get("state") == "harmful"):
            problems.append(
                f"envelope row {row.get('condition_id')}: a harmful row, and only a harmful row, names the "
                f"lines it failed, from {FAILED_LINES}"
            )
    labelled_rows = [
        r for r in label["envelope"].get("rows", []) if r.get("build_id") == labelled.get("build_id")
    ]
    lines = label["summary"].get("lines", [])
    own_state = {r.get("condition_id"): r.get("state") for r in labelled_rows}
    listed = []
    for line in lines:
        state, cause, conds = line.get("state"), line.get("cause"), line.get("conditions", [])
        if state not in STATES or (cause in HARM_CAUSES) != (state == "harmful"):
            problems.append(f"summary: line {state!r} / {cause!r} is not a known group")
        if line.get("count") is not None and line["count"] != len(conds):
            problems.append(f"summary: the {state!r} line's count differs from its conditions")
        if any(own_state.get(c) != state for c in conds):
            problems.append(f"summary: the {state!r} line lists a condition in another state")
        if state == "harmful":
            parts = line.get("by_failed", [])
            split = [c for part in parts for c in part.get("conditions", [])]
            counts_ok = all(p.get("count") == len(p.get("conditions", [])) for p in parts)
            if sorted(split) != sorted(conds) or not counts_ok:
                problems.append(f"summary: the {cause!r} line's split by failed line does not match it")
        listed += conds
    if sorted(listed) != sorted(own_state):
        problems.append("summary: every tested condition must be listed exactly once")
    reference_rows = [
        r for r in label["envelope"].get("rows", []) if r.get("build_id") != labelled.get("build_id")
    ]
    expected = {  # the three counts that open the label, each in row order
        "large_shrinking_cost": [
            r for r in labelled_rows if "large shrinking cost" in (r.get("shrinking_cost_flags") or [])
        ],
        "reference_harmful": [r for r in reference_rows if r.get("state") == "harmful"],
        "coverage_failed": [r for r in labelled_rows if "coverage" in r.get("failed", [])],
    }
    for name, expected_rows in expected.items():
        conds = [r["condition_id"] for r in expected_rows]
        if label["summary"].get(name) != {"count": len(conds), "conditions": conds}:
            problems.append(f"summary: {name} must count and name exactly the envelope's conditions")

    for s in label["speed"]:
        if s.get("status") not in SPEED_STATUSES or s.get("build_id") not in builds:
            problems.append(
                f"speed row {s.get('build_id')}: needs a known build and a status in {SPEED_STATUSES}"
            )
        elif s["status"] == "measured":
            where = f"speed row {s.get('build_id')} ({(s.get('settings') or {}).get('threads')} threads)"
            if (
                s.get("hardware_id") not in hardware
                or s.get("runtime_id") not in runtimes
                or not all(_finite(s.get(k)) for k in SPEED_FIELDS)
                or not isinstance(s.get("unstable"), bool)
            ):
                problems.append(
                    f"{where}: a measured row needs its hardware, runtime, {SPEED_FIELDS} and 'unstable'"
                )
            elif s["unstable"] != (s["spread_pct"] > s["unstable_above_pct"]):
                problems.append(f"{where}: 'unstable' must say whether the spread is above its line")
            ratio = s.get("time_vs_reference")
            if ratio is not None and (
                not _finite(ratio.get("ratio_p50")) or ratio.get("slower") != (ratio["ratio_p50"] > 1)
            ):
                problems.append(f"{where}: time_vs_reference needs a ratio, and 'slower' means above 1")
            problems += _check_sources(s.get("sources"), where)
        elif not s.get("reason") or s.get("runtime_id") is not None:
            problems.append(f"speed row {s.get('build_id')}: 'not measured' needs a reason and no runtime")
    if not label["sources"]:
        problems.append("the label lists no source files")
    else:
        problems += _check_sources(label["sources"], "sources")
    if source.get("kind") == USER_KIND:
        problems += _check_user_label(label)
    return problems


def study_label_problems(source_records) -> list:
    """The guard of decision 1 (H, 10 October 2026), read literally: no record from a user's run feeds a
    released study label. `source_records` are a label's source files already read as JSON; anything without
    a "source" field (a build record, the model list) is ignored. The site reads no records, and may show a
    user-submitted label marked unverified (website plan, step 2), so it needs no guard of its own."""
    if any(isinstance(r, dict) and r.get("source") == USER_KIND for r in source_records):
        return ["a released label must not be made from a user's run (a source record says 'user-submitted')"]
    return []


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def _check_fp32_sanity(s: dict, n_classes) -> list:
    """The expected-accuracy check as stored on a user label (decision 5); `pass` is worked out again here,
    with the rule of brokkr_edge.user_checks.expected_accuracy."""
    published = s.get("published")
    needed = ("measured", "n_items", "tolerance", "half_width_99", "allowed_gap", "chance")
    if not (isinstance(published, dict) and all(_finite(s.get(k)) for k in needed)
            and _finite(published.get("value")) and isinstance(s.get("ci95"), list) and len(s["ci95"]) == 2
            and isinstance(s.get("pass"), bool) and s.get("dataset_id")):
        return ["checks.fp32_sanity: a user label stores measured (with ci95, n_items and dataset_id), "
                "published, tolerance, half_width_99, allowed_gap, chance and pass"]
    problems = []
    if published.get("source") != "stated by the submitter" or not published.get("measured_on") or not (
        isinstance(published.get("n_images"), int) and published["n_images"] > 0
    ):
        problems.append("checks.fp32_sanity.published: the stated figure needs its source "
                        "('stated by the submitter'), its image count and where it was measured")
    p, n = published["value"], s["n_items"]
    expected = {
        "tolerance": 0.05,
        "half_width_99": Z99 * math.sqrt(p * (1 - p) / n) if n > 0 else math.inf,
        "chance": 1.0 / n_classes if _finite(n_classes) and n_classes > 0 else math.inf,
    }
    expected["allowed_gap"] = max(expected["tolerance"], expected["half_width_99"])
    for key, value in expected.items():
        if not math.isclose(s[key], value, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"checks.fp32_sanity.{key} differs from the rule (expected {value!r})")
    within = abs(s["measured"] - p) <= expected["allowed_gap"] + 1e-12
    if s["pass"] != (within and s["ci95"][0] > expected["chance"]):
        problems.append("checks.fp32_sanity.pass differs from the rule (within the allowed gap, and the "
                        "lower end of the 95% interval above chance)")
    return problems


def _check_warning(w) -> list:
    kind = w.get("kind") if isinstance(w, dict) else None
    if kind not in WARNING_FIELDS:
        return [f"summary.warnings: unknown kind {kind!r}; known: {tuple(WARNING_FIELDS)}"]
    missing = [k for k in WARNING_FIELDS[kind] if w.get(k) is None]
    if missing:
        return [f"summary.warnings {kind!r}: missing {missing}"]
    if not all(_finite(w[k]) for k in WARNING_FIELDS[kind] if k not in ("part", "classes")):
        return [f"summary.warnings {kind!r}: its counts must be numbers"]
    if kind == "few images" and not (w["part"] in ("test", "conformal_calibration")
                                     and w["n_items"] < w["recommended"]):
        return ["summary.warnings 'few images': a part (test or conformal_calibration) below its "
                "recommended size"]
    if kind == "classes with few test images" and not (isinstance(w["classes"], list) and w["classes"]):
        return ["summary.warnings 'classes with few test images': name at least one class"]
    if kind == "low agreement with FP32" and not w["agreement"] < w["warn_below"]:
        return ["summary.warnings 'low agreement with FP32': the agreement must be below its line"]
    if kind == "few usable reference predictions" and not w["usable"] < w["below"]:
        return ["summary.warnings 'few usable reference predictions': the count must be below its line"]
    return []


def _check_user_label(label: dict) -> list:
    """The rules a user label follows on top of schema version 1 (see the module docstring)."""
    problems = []
    model, source, generated = label["model"], label["source"], label["generated"]
    reference = next((b for b in label["builds"] if b.get("role") == "reference"), {})
    labelled = next((b for b in label["builds"] if b.get("role") == "labelled"), {})
    found = USER_MODEL_ID.match(str(model.get("model_id", "")))
    if model.get("publisher") != "user" or not found:
        problems.append("a user label's model.publisher is 'user' and its model_id user/<name>@<12 hex>")
    elif found.group(2) != str((reference.get("file") or {}).get("sha256", ""))[:12]:
        problems.append("a user label's model_id ends with the first 12 hex characters of the FP32 file's "
                        "SHA-256")
    if not str(source.get("how_made", "")).startswith("brokkr-edge test"):
        problems.append("a user label's source.how_made is 'brokkr-edge test <version>'")
    if not (source.get("submitted_by") is None or isinstance(source.get("submitted_by"), str)):
        problems.append("source.submitted_by is a name or null")

    if generated.get("seed_scheme") != USER_SEED_SCHEME:
        problems.append(f"a user label's generated.seed_scheme is {USER_SEED_SCHEME!r} (decision 4)")
    commit, dirty = generated.get("commit"), generated.get("dirty")
    if commit is None:
        if dirty is not None:
            problems.append("outside a git checkout, generated.commit and generated.dirty are both null")
    elif not (isinstance(commit, str) and len(commit) == 40 and isinstance(dirty, bool)):
        problems.append("generated.commit is a 40-character commit with a dirty flag, or null outside git")

    name = found.group(1) if found else None
    for d in label["datasets"]:
        where = f"dataset {d.get('dataset_id')}"
        if name and not str(d.get("dataset_id", "")).startswith(f"user/{name}:"):
            problems.append(f"{where}: a user dataset's ID is user/{name}:<part>")
        if d.get("split_mode") not in SPLIT_MODES:
            problems.append(f"{where}: split_mode must be one of {SPLIT_MODES}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(d.get("folder_fingerprint", ""))):
            problems.append(f"{where}: needs its folder_fingerprint (a SHA-256)")

    licences = label["licences"]
    if not licences.get("images") or licences.get("declared_by_submitter") is not True:
        problems.append("a user label's licences hold the images licence, with declared_by_submitter: true")
    if (label.get("declarations") or {}).get("images_not_used_to_train_the_model") is not True:
        problems.append("a user label declares images_not_used_to_train_the_model: true (D10)")

    sanity = label["checks"].get("fp32_sanity")
    if not isinstance(sanity, dict):
        problems.append("a user label has checks.fp32_sanity")
    else:
        problems += _check_fp32_sanity(sanity, (model.get("outputs") or {}).get("n_classes"))

    warnings = label["summary"].get("warnings", [])
    if not isinstance(warnings, list):
        problems.append("summary.warnings is a list")
    else:
        for w in warnings:
            problems += _check_warning(w)

    recipe = labelled.get("recipe") or {}
    supplied_ok = recipe.get("made_by") and isinstance(recipe.get("found"), dict)
    if recipe.get("method") == SUPPLIED and not supplied_ok:
        problems.append("a supplied build's recipe names who made it (made_by) and what was found in it")

    hardware_ids = [h.get("hardware_id") for h in label["hardware"]]
    if len(hardware_ids) != 1:
        problems.append("a user label names exactly one machine: the one that ran the test")
    rows = label["speed"]
    if sorted(s.get("build_id") for s in rows) != sorted(b.get("build_id") for b in label["builds"]) or any(
        s.get("status") != "not measured" or s.get("reason") != USER_SPEED_REASON
        or s.get("hardware_id") not in hardware_ids
        for s in rows
    ):
        problems.append(f"a user label has one 'not measured' speed row per build, on its own machine, with "
                        f"the reason {USER_SPEED_REASON!r} (decision 8)")

    paths = sorted({s for s in _strings(label) if MACHINE_PATH.search(s)})
    if paths:
        problems.append(f"a user label holds no paths from the user's machine: {paths[:3]}")
    return problems
