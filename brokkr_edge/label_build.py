"""Building a label from numbers that are already read and checked (docs/label_schema.md, schema version 1).

Shared by the study's label maker (scripts/39_make_labels.py) and the user path (docs/user_models.md, note
of 10 October 2026). Nothing here reads a record: callers find and check their records, then pass in the
numbers, the per-image right/wrong arrays and the `sources` entries that name where each number came from.
So a label is put together the same way whoever made the measurements.

- measurement(): one entry of `measurements` (copied from a record, or derived).
- derived_measurements(): damage drop and shrinking cost, paired (brokkr_edge.label.paired).
- envelope_rows(): each build's state in each damaged condition (brokkr_edge.label's rules).
- general_limits(): the plain sentences every label carries.
- assemble(): the label's top-level fields, in schema order, with the summary made from the envelope.
"""

import datetime
import re
from pathlib import Path

from brokkr_edge import __version__
from brokkr_edge.label import (
    CI_LEVEL,
    ENVELOPE_RULE,
    envelope_state,
    failed_lines,
    line_states,
    paired,
    shrinking_cost_flags,
    summary,
)
from brokkr_edge.label_schema import condition_id, runtime_id
from brokkr_edge.schema import condition_label

ROLES = ("reference", "labelled")
# The study's sentence about where accuracy was measured (the user path has its own; note of 10 October 2026).
STUDY_MACHINE_SENTENCE = "Accuracy was measured on the laptop; it has not been checked on other hardware."


def imagecorruptions_version() -> str:
    """The vendored imagecorruptions copy's version, from the first line of its notes ("# Vendored copy of
    imagecorruptions 1.1.2"), for the ImageNet-C limit (H's fix list, 4 October 2026, item 1)."""
    notes = Path(__file__).parent / "third_party" / "imagecorruptions" / "CHANGES.md"
    return re.match(r"# Vendored copy of imagecorruptions (\d+\.\d+\.\d+)\n",
                    notes.read_text(encoding="utf-8")).group(1)


def runtime_entry(runtime: dict, sources: list) -> dict:
    """One `runtimes` entry from a record's runtime block (docs/label_schema.md, note of 3 October 2026)."""
    return {
        "runtime_id": runtime_id(runtime["name"], runtime["version"], runtime["execution_provider"],
                                 runtime["threads"], runtime.get("spinning")),
        "name": runtime["name"],
        "version": runtime["version"],
        "execution_provider": runtime["execution_provider"],
        "threads": runtime["threads"],
        "intra_op_threads": runtime.get("intra_op_threads"),
        "inter_op_threads": runtime.get("inter_op_threads"),
        "spinning": runtime.get("spinning"),
        "graph_optimisation": runtime.get("graph_optimisation"),
        "sources": sources,
    }


def conditions_block(conditions: list) -> list:
    """The label's `conditions` from schema conditions ({"corruption", "suite", "severity"}), in order."""
    return [
        {
            "condition_id": condition_id(c["suite"], c["corruption"], c["severity"]),
            "suite": c["suite"],
            "damage": c["corruption"],
            "severity": c["severity"] or None,
            "modality": "image",
            "label": condition_label(c),
        }
        for c in conditions
    ]


def failure_block(check: str, value, limit, n_items, split, sources: list) -> dict:
    """Why a build failed (docs/label_schema.md, note of 30 September 2026, point 3)."""
    return {"check": check, "value": value, "limit": limit, "n_items": n_items, "split": split,
            "sources": sources}


def build_entry(build_id: str, role: str, display_name: str, precision: str, recipe: dict, sha256: str,
                size_bytes: int, status: str, failure: dict | None) -> dict:
    """One `builds` entry."""
    return {
        "build_id": build_id,
        "role": role,
        "display_name": display_name,
        "precision": precision,
        "recipe": recipe,
        "file": {"sha256": sha256, "size_bytes": size_bytes},
        "status": status,
        "failure": failure,
    }


def measurement(metric: str, build_id: str, dataset_id: str, cond_id: str, hardware_id: str, runtime: str,
                value, ci95, n_items: int, settings: dict, sources: list,
                paired_runtime: str | None = None) -> dict:
    """One `measurements` entry. A derived one whose second record ran with another runtime names that runtime
    in settings.paired_runtime_id (note of 3 October 2026)."""
    if paired_runtime and paired_runtime != runtime:
        settings = {**settings, "paired_runtime_id": paired_runtime}
    return {
        "metric": metric,
        "build_id": build_id,
        "dataset_id": dataset_id,
        "condition_id": cond_id,
        "hardware_id": hardware_id,
        "runtime_id": runtime,
        "value": value,
        "ci95": ci95,
        "n_items": n_items,
        "unit": "classes" if metric == "mean_set_size" else "fraction",
        "settings": settings,
        "sources": sources,
    }


def derived_measurements(builds: dict, status: dict, condition_ids: list, correct: dict, score_sources: dict,
                         runtimes: dict, dataset_id: str, hardware_id: str) -> list:
    """Damage drop (each usable build, each damaged condition, against its own clean run), then shrinking cost
    (labelled minus reference, every condition, if the labelled build is usable). Both paired over the same
    images.

    builds, status: by role. condition_ids: clean first, then the damaged ones. correct, score_sources,
    runtimes: by (role, condition ID): the per-image 0/1 array, the source entry of its score file, its
    runtime ID.
    """
    out = []
    for role in ROLES:
        if status[role] != "usable":
            continue
        for cid in condition_ids[1:]:
            p = paired(correct[(role, cid)], correct[(role, "clean")])
            out.append(measurement(
                "damage_drop", builds[role]["build_id"], dataset_id, cid, hardware_id, runtimes[(role, cid)],
                p["value"], p["ci95"], p["n_items"], {"paired_with": "clean", "bootstrap": p["bootstrap"]},
                [score_sources[(role, cid)], score_sources[(role, "clean")]],
                paired_runtime=runtimes[(role, "clean")]))
    if status["labelled"] == "usable":
        for cid in condition_ids:
            p = paired(correct[("labelled", cid)], correct[("reference", cid)])
            out.append(measurement(
                "shrinking_cost", builds["labelled"]["build_id"], dataset_id, cid, hardware_id,
                runtimes[("labelled", cid)], p["value"], p["ci95"], p["n_items"],
                {"paired_with": builds["reference"]["build_id"], "bootstrap": p["bootstrap"]},
                [score_sources[("labelled", cid)], score_sources[("reference", cid)]],
                paired_runtime=runtimes[("reference", cid)]))
    return out


def envelope_rows(builds: dict, status: dict, damaged_ids: list, measurements: list,
                  hardware_id: str) -> list:
    """Each build's envelope row in each damaged condition (reference first), from its coverage and
    damage-drop intervals; a failed build's rows say "INT8 build failed"."""
    index = {(m["metric"], m["build_id"], m["condition_id"]): m for m in measurements}
    rows = []
    for role in ROLES:
        bid = builds[role]["build_id"]
        for cid in damaged_ids:
            if status[role] != "usable":
                rows.append({
                    "build_id": bid,
                    "condition_id": cid,
                    "hardware_id": hardware_id,
                    "state": "INT8 build failed",
                    "why": [f"build check failed: {builds[role]['failure']['check']}"],
                    "failed": [],
                    "line_states": None,
                    "shrinking_cost_flags": [],
                })
                continue
            coverage_ci = index[("coverage", bid, cid)]["ci95"]
            drop_ci = index[("damage_drop", bid, cid)]["ci95"]
            state, why = envelope_state(coverage_ci, drop_ci)
            flags = []
            if role == "labelled":
                fp32_top1 = index[("top1", builds["reference"]["build_id"], cid)]["value"]
                flags = shrinking_cost_flags(index[("shrinking_cost", bid, cid)]["ci95"], fp32_top1)
            rows.append({
                "build_id": bid,
                "condition_id": cid,
                "hardware_id": hardware_id,
                "state": state,
                "why": why,
                "failed": failed_lines(coverage_ci, drop_ci),
                "line_states": line_states(coverage_ci, drop_ci),
                "shrinking_cost_flags": flags,
            })
    return rows


def general_limits(cpu_model: str, os_name: str, n_damaged: int, machine_sentence: str) -> list:
    """The plain sentences every label carries, in order; machine_sentence says where accuracy was
    measured."""
    return [
        "The damage is simulated (Brokkr's own and ImageNet-C corruptions); real fog, darkness or noise may "
        "affect the model differently.",
        "The ImageNet-C conditions were made on these test images with the imagecorruptions package "
        f"v{imagecorruptions_version()} (an extension of the ImageNet-C code), with a one-line fix so that "
        "fog runs on NumPy 2, tested pixel-identical to the unmodified package's fog. Fixed seeds and each "
        "model's own preprocessing were used, so the results are not directly comparable to the released "
        "ImageNet-C files or to published ImageNet-C results.",
        f"Measured on one machine: {cpu_model}, {os_name}.",
        f"With {n_damaged} conditions, an occasional result may cross a line by chance.",
        "Coverage is for prediction sets tuned on clean calibration images; under damage there is no "
        "coverage promise, and the label shows what was measured.",
        machine_sentence,
        "The suggested next steps are general suggestions; they were not tested for this model.",
    ]


def generated_block(commit, dirty, label_licence: str) -> dict:
    """Which code made the label, and when."""
    return {
        "by": f"brokkr-edge {__version__}",
        "commit": commit,
        "dirty": dirty,
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "ci_level": CI_LEVEL,
        "label_licence": label_licence,
        "envelope_rule": ENVELOPE_RULE["name"],
    }


def assemble(*, source: dict, generated: dict, model: dict, builds: dict, hardware: list, runtimes: list,
             datasets: list, conditions: list, checks: dict, measurements: list, rows: list, speed: list,
             details: dict, limits: list, licences: dict, sources: list) -> dict:
    """The label, top-level fields in schema order; the summary is made from the envelope rows."""
    return {
        "schema_version": 1,
        "label_id": builds["labelled"]["build_id"],
        "source": source,
        "generated": generated,
        "model": model,
        "builds": [builds["reference"], builds["labelled"]],
        "hardware": hardware,
        "runtimes": runtimes,
        "datasets": datasets,
        "conditions": conditions,
        "checks": checks,
        "measurements": measurements,
        "envelope": {"rule": ENVELOPE_RULE, "rows": rows},
        "summary": summary(rows, builds["labelled"]["build_id"]),
        "speed": speed,
        "details": details,
        "limits": limits,
        "licences": licences,
        "sources": sources,
    }
