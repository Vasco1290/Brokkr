"""Reading the released labels for the website: lookups, build IDs and page addresses.

Used by the site generator (web/site_build.py) and the design prototypes. Reads only label.json data;
computes no number. Build IDs and addresses follow docs/website_v0_plan.md, section 12d (H, 4 October 2026).
"""

import json
import re
from pathlib import Path

from brokkr_edge.label_schema import SCHEMA_VERSION, check_label

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "published" / "labels"
SOURCE_WORDS = {"official": "Brokkr study", "user-submitted": "User-submitted"}  # H, 3 October 2026


def load_labels(folder: Path = LABELS) -> dict:
    """Every label.json in `folder`, keyed by its folder name, sorted by display name."""
    labs = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*/label.json")}
    return dict(sorted(labs.items(), key=lambda kv: kv[1]["model"]["display_name"].lower()))


def label_problems(labs: dict) -> list:
    """Schema problems of every label; a schema version the site does not know is refused, not guessed."""
    problems = []
    for name, lab in labs.items():
        if lab.get("schema_version") != SCHEMA_VERSION:
            problems.append(f"{name}: label schema version {lab.get('schema_version')!r} is not known here")
            continue
        problems += [f"{name}: {p}" for p in check_label(lab)]
    return problems


def builds(label: dict) -> tuple:
    """(reference build, labelled build)."""
    by_role = {b["role"]: b for b in label["builds"]}
    return by_role["reference"], by_role["labelled"]


def measurement(label: dict, metric: str, build: dict, cid: str):
    return next(
        (
            m
            for m in label["measurements"]
            if (m["metric"], m["build_id"], m["condition_id"]) == (metric, build["build_id"], cid)
        ),
        None,
    )


def envelope_row(label: dict, build: dict, cid: str):
    """The envelope row of a build in a damaged condition; None for clean (the reference, not judged)."""
    return next(
        (
            r
            for r in label["envelope"]["rows"]
            if (r["build_id"], r["condition_id"]) == (build["build_id"], cid)
        ),
        None,
    )


def condition_names(label: dict) -> dict:
    return {c["condition_id"]: c["label"] for c in label["conditions"]}


def the_test_split(label: dict) -> dict:
    """The dataset entry of the test split (named so pytest never mistakes it for a test)."""
    return next(d for d in label["datasets"] if d["split"] == "test")


# ---- build IDs and addresses (section 12d) ----

PART = re.compile(r"[A-Z]+[0-9.]*(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+[0-9.]*|[0-9.]+[A-Za-z]*")


def model_code(display_name: str) -> str:
    """'MobileNetV3-Large' -> 'MNV3L': split at spaces, hyphens and capitals; a plain word gives its first
    letter, a part with digits or in capitals is kept whole (dots dropped)."""
    code = ""
    for word in re.split(r"[\s-]+", display_name):
        for part in PART.findall(word):
            whole = any(c.isdigit() for c in part) or part.isupper()
            code += part.upper().replace(".", "") if whole else part[0].upper()
    return code


def build_slug(build: dict) -> str:
    """'int8' + 'percentile99.99' -> 'int8-percentile-99.99'; FP32 -> 'fp32'."""
    method = (build.get("recipe") or {}).get("method") or ""
    m = re.fullmatch(r"([a-z]+)([0-9.]*)", method)
    parts = [build["precision"], *(p for p in (m.groups() if m else ()) if p)]
    return "-".join(parts)


def build_id(label: dict, build: dict) -> str:
    """'BRK-' + model code + precision + recipe code (first letter of the method, then its digits)."""
    slug = build_slug(build).split("-")
    recipe = (
        (slug[1][0].upper() + (slug[2].replace(".", "") if len(slug) > 2 else "")) if len(slug) > 1 else ""
    )
    return "-".join(
        p for p in ("BRK", model_code(label["model"]["display_name"]), slug[0].upper(), recipe) if p
    )


def model_slug(label: dict) -> str:
    return label["model"]["name"].replace("_", "-")


def label_dir(label: dict) -> str:
    """The folder of a label's page, e.g. 'models/mobilenet-v3-large/int8-percentile-99.99'."""
    return f"models/{model_slug(label)}/{build_slug(builds(label)[1])}"


def address(label: dict) -> str:
    """The page's public address, e.g. '/models/mobilenet-v3-large/int8-percentile-99.99/'."""
    return f"/{label_dir(label)}/"
