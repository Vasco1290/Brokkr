"""The label format (docs/label_schema.md, schema version 1): stable IDs and a validator.

check_label(label) returns every problem as a plain sentence; an empty list means the label follows
schema version 1. It checks structure and consistency only; whether each number equals its source
file is checked by scripts/40_check_labels.py.
"""

import math
import re

SCHEMA_VERSION = 1
SOURCE_KINDS = ("official", "user-submitted")
MODALITIES = ("image", "signal")
DEVICE_KINDS = ("laptop", "raspberry-pi-5", "cloud-arm")
BUILD_ROLES = ("reference", "labelled")
BUILD_STATUSES = ("usable", "failed")
STATES = ("not harmful", "harmful", "borderline", "not tested", "INT8 build failed")
SHRINKING_COST_FLAGS = (None, "large shrinking cost", "not informative")
# Summary groups (note of 30 September 2026 in docs/label_schema.md; the rule is brokkr_edge.label).
SUMMARY_GROUPS = (
    "fine",
    "too hard for this model",
    "hurt by shrinking",
    "borderline",
    "INT8 build failed",
    "not tested",
)
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
    "datasets",
    "conditions",
    "checks",
    "measurements",
    "envelope",
    "summary",
    "speed",
    "details",
    "limits",
    "sources",
)
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.:/@#-]*$")
ABSOLUTE_PATH = re.compile(r"^([A-Za-z]:)?[/\\]")  # a path from this machine: "C:/...", "/home/..."


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
    for h in label["hardware"]:
        if h.get("kind") not in DEVICE_KINDS or not ID_PATTERN.match(str(h.get("hardware_id", ""))):
            problems.append(
                f"hardware {h.get('hardware_id')!r}: needs a valid ID and a kind in {DEVICE_KINDS}"
            )
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
        ):
            problems.append(f"{where}: refers to a build, dataset, condition or hardware not in the label")
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
        if row.get("shrinking_cost_flag") not in SHRINKING_COST_FLAGS:
            problems.append(f"envelope row {row.get('condition_id')}: unknown flag")
        if row.get("build_id") not in builds or row.get("condition_id") not in conditions:
            problems.append(
                f"envelope row {row.get('condition_id')}: refers to an unknown build or condition"
            )
        elif builds[row["build_id"]].get("status") == "failed" and row.get("state") != "INT8 build failed":
            problems.append(
                f"envelope row {row.get('condition_id')}: a failed build's rows say 'INT8 build failed'"
            )
    labelled_rows = [
        r for r in label["envelope"].get("rows", []) if r.get("build_id") == labelled.get("build_id")
    ]
    lines = label["summary"].get("lines", [])
    unknown = [line.get("group") for line in lines if line.get("group") not in SUMMARY_GROUPS]
    if unknown:
        problems.append(f"summary: unknown groups {unknown}")
    if sum(line.get("count") or 0 for line in lines) != len(labelled_rows):
        problems.append("summary: its counts do not add up to the labelled build's envelope rows")

    for s in label["speed"]:
        if s.get("status") not in SPEED_STATUSES or s.get("build_id") not in builds:
            problems.append(
                f"speed row {s.get('build_id')}: needs a known build and a status in {SPEED_STATUSES}"
            )
        elif s["status"] == "measured":
            if s.get("hardware_id") not in hardware or not all(
                _finite(s.get(k)) for k in ("p50_ms", "p95_ms", "p99_ms")
            ):
                problems.append(
                    f"speed row {s.get('build_id')}: a measured row needs its hardware and p50/p95/p99"
                )
            problems += _check_sources(s.get("sources"), f"speed row {s.get('build_id')}")
        elif not s.get("reason"):
            problems.append(f"speed row {s.get('build_id')}: 'not measured' needs a reason")
    if not label["sources"]:
        problems.append("the label lists no source files")
    else:
        problems += _check_sources(label["sources"], "sources")
    return problems
