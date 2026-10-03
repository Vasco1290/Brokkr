"""The rules that turn measurements into a label (docs/label_schema.md; decisions in
docs/hypotheses_stage4.md, notes of 29-30 September 2026). Nothing here reads files: scripts gather
the checked records and pass the numbers in, so the same rules serve images, signals and any device.

- Damage drop = top-1 under a condition minus clean top-1, same build, same items.
- Shrinking cost = labelled build minus reference build, same condition, same items.
  Both are paired: each bootstrap resample of the items computes both sides (brokkr_edge.judge.paired_ci:
  1,000 resamples, seed 0); the value is counted in whole items.
- Envelope state of a build in a condition, from its coverage interval and damage-drop interval:
  "not harmful" if coverage's lower end >= 80% and the drop's lower end >= -10 points;
  "harmful" if coverage's upper end < 80% or the drop's upper end < -10 points; otherwise "borderline".
- Shrinking-cost flags, each that applies (note of 3 October 2026: one never hides the other):
  "large shrinking cost" if the whole interval is below -5 points; "not informative" if FP32 top-1
  under the condition is below 10%.
- Failed lines of a harmful row: which whole interval is below its line ("damage drop", "coverage").
- Summary (docs/label_schema.md, second note of 30 September 2026): the labelled build's conditions by
  its own state, harmful ones by cause (see harm_cause()), and every large shrinking cost named.
"""

import numpy as np

from brokkr_edge.judge import N_RESAMPLES, SEED, paired_ci
from brokkr_edge.label_schema import HARM_CAUSES

ENVELOPE_RULE = {
    "name": "reliability envelope, version 1",
    "fixed_in": "docs/hypotheses_stage4.md, notes of 29 and 30 September 2026",
    "harm_definition": "the harm definition of 621cc43 (3.9 note), applied to the test split",
    "coverage_min": 0.80,
    "damage_drop_min": -0.10,
    "large_shrinking_cost_below": -0.05,
    "near_floor_fp32_top1_below": 0.10,
}
CI_LEVEL = 0.95


def paired(new_correct: np.ndarray, old_correct: np.ndarray) -> dict:
    """New minus old (per-item 0/1 arrays over the same items), with its paired 95% interval."""
    if new_correct.shape != old_correct.shape:
        raise ValueError("a paired comparison needs the same items on both sides")
    per_item = new_correct - old_correct
    count = int(new_correct.sum() - old_correct.sum())
    ci = paired_ci(lambda idx: per_item[idx].mean(), len(per_item))
    return {
        "value": count / len(per_item),
        "ci95": ci,
        "n_items": len(per_item),
        "count": count,
        "bootstrap": {"resamples": N_RESAMPLES, "seed": SEED, "paired": True},
    }


def failed_lines(coverage_ci: list, drop_ci: list, rule: dict = ENVELOPE_RULE) -> list:
    """The lines whose whole interval is below them: a subset of ["damage drop", "coverage"]."""
    failed = []
    if drop_ci[1] < rule["damage_drop_min"]:
        failed.append("damage drop")
    if coverage_ci[1] < rule["coverage_min"]:
        failed.append("coverage")
    return failed


def line_states(coverage_ci: list, drop_ci: list, rule: dict = ENVELOPE_RULE) -> dict:
    """The row's state on each line (note of 3 October 2026, later the same day): "fails" (whole interval
    below the line), "copes" (whole interval at or above it) or "straddles"."""
    def one(ci, line):
        return "fails" if ci[1] < line else "copes" if ci[0] >= line else "straddles"

    return {"damage drop": one(drop_ci, rule["damage_drop_min"]),
            "coverage": one(coverage_ci, rule["coverage_min"])}


def envelope_state(coverage_ci: list, drop_ci: list, rule: dict = ENVELOPE_RULE) -> tuple:
    """(state, reasons) for one build in one condition."""
    cov_min, drop_min = rule["coverage_min"], rule["damage_drop_min"]
    failed = failed_lines(coverage_ci, drop_ci, rule)
    if failed:
        why = {"damage drop": "whole damage-drop interval below the damage-drop line",
               "coverage": "whole coverage interval below the coverage line"}
        return "harmful", [why[name] for name in failed]
    clears = coverage_ci[0] >= cov_min and drop_ci[0] >= drop_min
    if clears:
        return "not harmful", [
            "whole coverage interval at or above the coverage line",
            "whole damage-drop interval at or above the damage-drop line",
        ]
    straddles = [
        name
        for name, ci, line in (("coverage", coverage_ci, cov_min), ("damage drop", drop_ci, drop_min))
        if ci[0] < line
    ]
    return "borderline", [f"{name} interval straddles its line" for name in straddles]


def shrinking_cost_flags(cost_ci: list, fp32_top1: float, rule: dict = ENVELOPE_RULE) -> list:
    """Every flag that applies, in a fixed order; both when both apply, so neither hides the other."""
    flags = []
    if cost_ci[1] < rule["large_shrinking_cost_below"]:
        flags.append("large shrinking cost")
    if fp32_top1 < rule["near_floor_fp32_top1_below"]:
        flags.append("not informative")
    return flags


def harm_cause(reference_state: str) -> str:
    """Why the labelled build is harmful in a condition, read from FP32's state there (no new threshold).

    too hard for this model: FP32 is harmful too (FP32 also fails).
    hurt by shrinking: FP32 is not harmful (FP32 copes, INT8 doesn't).
    cause unclear: FP32 is borderline, so it neither clearly copes nor clearly fails.
    """
    return {"harmful": "too hard for this model", "not harmful": "hurt by shrinking"}.get(
        reference_state, "cause unclear"
    )


def _counted(conditions: list) -> dict:
    return {"count": len(conditions), "conditions": conditions}


def summary(rows: list, labelled_build_id: str) -> dict:
    """The labelled build's conditions grouped by its own state, then harmful ones by cause (row order
    kept), and the three counts that open the label: conditions with a large shrinking cost (whatever
    their group), conditions where FP32 itself is harmful, and conditions that fail the coverage line."""
    reference = {r["condition_id"]: r["state"] for r in rows if r["build_id"] != labelled_build_id}
    own = [r for r in rows if r["build_id"] == labelled_build_id]
    order = [(state, None) for state in ("not harmful", "borderline")]
    order += [("harmful", cause) for cause in HARM_CAUSES]
    order += [("INT8 build failed", None)]
    grouped = {key: [] for key in order}
    for r in own:
        cause = harm_cause(reference[r["condition_id"]]) if r["state"] == "harmful" else None
        grouped[(r["state"], cause)].append(r)
    lines = []
    for (state, cause), group in grouped.items():
        if not group:
            continue
        line = {"state": state, "cause": cause, "count": len(group),
                "conditions": [r["condition_id"] for r in group]}
        if state == "harmful":  # which line each condition failed: accuracy, coverage, or both
            line["by_failed"] = [
                {"failed": failed, "count": len(conds), "conditions": conds}
                for failed in (["damage drop"], ["coverage"], ["damage drop", "coverage"])
                if (conds := [r["condition_id"] for r in group if r["failed"] == failed])
            ]
        lines.append(line)
    lines.append({"state": "not tested", "cause": None, "count": None, "conditions": []})
    return {
        "describes": labelled_build_id,
        "tested_conditions": len(own),
        "lines": lines,
        "large_shrinking_cost": _counted(
            [r["condition_id"] for r in own if "large shrinking cost" in r.get("shrinking_cost_flags", [])]
        ),
        "reference_harmful": _counted(
            [
                r["condition_id"]
                for r in rows
                if r["build_id"] != labelled_build_id and r["state"] == "harmful"
            ]
        ),
        "coverage_failed": _counted([r["condition_id"] for r in own if "coverage" in r["failed"]]),
    }
