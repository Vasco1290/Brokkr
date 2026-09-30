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
- Shrinking-cost flag: "not informative" if FP32 top-1 under the condition is below 10%; else "large
  shrinking cost" if the whole interval is below -5 points.
- Summary group of a condition, from the labelled build's state and the reference build's state there
  (docs/label_schema.md, note of 30 September 2026): see summary_group().
"""

import numpy as np

from brokkr_edge.judge import N_RESAMPLES, SEED, paired_ci
from brokkr_edge.label_schema import SUMMARY_GROUPS

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


def envelope_state(coverage_ci: list, drop_ci: list, rule: dict = ENVELOPE_RULE) -> tuple:
    """(state, reasons) for one build in one condition."""
    cov_min, drop_min = rule["coverage_min"], rule["damage_drop_min"]
    fails = []
    if coverage_ci[1] < cov_min:
        fails.append("whole coverage interval below the coverage line")
    if drop_ci[1] < drop_min:
        fails.append("whole damage-drop interval below the damage-drop line")
    if fails:
        return "harmful", fails
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


def shrinking_cost_flag(cost_ci: list, fp32_top1: float, rule: dict = ENVELOPE_RULE) -> str | None:
    if fp32_top1 < rule["near_floor_fp32_top1_below"]:
        return "not informative"
    if cost_ci[1] < rule["large_shrinking_cost_below"]:
        return "large shrinking cost"
    return None


JUDGED_GROUPS = SUMMARY_GROUPS[:-1]  # every group except "not tested", in the order shown


def summary_group(labelled_state: str, reference_state: str) -> str:
    """Which summary group a condition goes in. Only the envelope states are used; no new threshold.

    fine: the labelled build is not harmful there (whatever FP32 does).
    too hard for this model: both builds are harmful (FP32 also fails).
    hurt by shrinking: the labelled build is harmful and FP32 is not harmful (FP32 copes, INT8 doesn't).
    borderline: the labelled build is borderline, or it is harmful while FP32 is borderline.
    """
    if labelled_state in ("not harmful", "INT8 build failed"):
        return "fine" if labelled_state == "not harmful" else labelled_state
    if labelled_state == "harmful" and reference_state == "harmful":
        return "too hard for this model"
    if labelled_state == "harmful" and reference_state == "not harmful":
        return "hurt by shrinking"
    return "borderline"


def summary(rows: list, labelled_build_id: str, condition_labels: dict) -> dict:
    """One line per summary group for the labelled build, each with its count and conditions, in row
    order. A harmful condition in "borderline" (because FP32 is borderline there) says so."""
    state = {(r["build_id"], r["condition_id"]): r["state"] for r in rows}
    reference = {r["condition_id"]: r["state"] for r in rows if r["build_id"] != labelled_build_id}
    grouped = {group: [] for group in JUDGED_GROUPS}
    for r in rows:
        if r["build_id"] != labelled_build_id:
            continue
        cid = r["condition_id"]
        group = summary_group(state[(labelled_build_id, cid)], reference[cid])
        name = condition_labels[cid]
        if group == "borderline" and r["state"] == "harmful":
            name += f" (INT8 harmful, FP32 {reference[cid]})"
        grouped[group].append(name)
    lines = [
        {"group": group, "count": len(conds), "conditions": conds}
        for group, conds in grouped.items()
        if conds
    ]
    lines.append(
        {
            "group": "not tested",
            "count": None,
            "conditions": ["every damage type and severity not listed above"],
        }
    )
    return {
        "describes": labelled_build_id,
        "tested_conditions": len([r for r in rows if r["build_id"] == labelled_build_id]),
        "lines": lines,
    }
