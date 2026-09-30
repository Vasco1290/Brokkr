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
"""

import numpy as np

from brokkr_edge.judge import N_RESAMPLES, SEED, paired_ci

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


def summary(rows: list, labelled_build_id: str, condition_labels: dict) -> dict:
    """One line per state for the labelled build, each with its count and conditions, in row order."""
    lines = []
    for state in ("not harmful", "borderline", "harmful", "INT8 build failed"):
        conds = [
            condition_labels[r["condition_id"]]
            for r in rows
            if r["build_id"] == labelled_build_id and r["state"] == state
        ]
        if conds:
            lines.append({"state": state, "count": len(conds), "conditions": conds})
    lines.append(
        {
            "state": "not tested",
            "count": None,
            "conditions": ["every damage type and severity not listed above"],
        }
    )
    return {
        "describes": labelled_build_id,
        "tested_conditions": len([r for r in rows if r["build_id"] == labelled_build_id]),
        "lines": lines,
    }
