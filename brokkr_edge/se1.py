"""The SE1 pass rule (docs/hypotheses_stage4.md: "Squeeze-and-excitation diagnostic (SE1)" and H's
confirmations of 27 September 2026), from per-image correctness (0/1 arrays over the same images).

Words:
- extra gap of a build under a condition: (build - FP32 top-1 under the condition) - (build - FP32
  top-1 on clean images), paired over the same images (brokkr_edge.judge_breadth.extra_gap);
- change of a variant: its extra gap minus the baseline build's (new minus old; positive = the variant
  loses less to the damage). Per image: (variant damaged - variant clean) - (baseline damaged -
  baseline clean), so FP32 cancels. Paired 95% interval: 1,000 resamples, seed 0;
- recovered share: change / (- baseline extra gap).
Every threshold is compared in whole images, so no rounding can move a value across a line.
"""

from brokkr_edge.judge import ci_side, paired_ci
from brokkr_edge.judge_breadth import extra_gap

BASELINE_LOSS_POINTS = 2   # a baseline loss: extra gap -2.0 points or lower, interval below zero
RECOVERED_SHARE = 0.5      # judged models: recover at least half
CONTROL_POINTS = 2         # control: |change| smaller than 2.0 points


def change(base_clean, base_cond, variant_clean, variant_cond) -> dict:
    """The variant's extra gap minus the baseline's, with its paired interval."""
    per_image = (variant_cond - variant_clean) - (base_cond - base_clean)
    count = int(variant_cond.sum() - variant_clean.sum() - base_cond.sum() + base_clean.sum())
    ci = paired_ci(lambda idx: per_image[idx].mean(), len(per_image))
    return {"value": count / len(per_image), "count": count, "n": len(per_image), "ci95": ci,
            "side": ci_side(ci)}


def baseline_extra_gap(fp32_clean, fp32_cond, base_clean, base_cond) -> dict:
    e = extra_gap(fp32_clean, base_clean, fp32_cond, base_cond)
    return {k: v for k, v in e.items() if k != "per_image"}


def has_baseline_loss(e: dict) -> bool:
    """At least 2.0 points lost to the damage (count / n <= -2 / 100), with the interval below zero."""
    return e["count"] * 100 <= -BASELINE_LOSS_POINTS * e["n"] and e["side"] == "below zero"


def share(baseline: dict, ch: dict) -> float | None:
    return ch["count"] / -baseline["count"] if baseline["count"] < 0 else None


def recovers(baseline: dict, ch: dict) -> bool:
    """Recovered share >= 50% (2 x change >= -baseline, in images) and the change's interval above zero."""
    return baseline["count"] < 0 and 2 * ch["count"] >= -baseline["count"] and ch["side"] == "above zero"


def control_holds(ch: dict) -> bool:
    """|change| < 2.0 points (|count| / n < 2 / 100)."""
    return abs(ch["count"]) * 100 < CONTROL_POINTS * ch["n"]


def verdict(judged: dict, control: dict) -> dict:
    """judged: model -> {"baseline": extra gap, "change": change of SE-all, "builds_ok": bool};
    control: {"change": ..., "builds_ok": bool}. All under darkness (Brokkr) s5."""
    parts = {}
    not_judged = [f"{m}: a build failed a build check" for m, d in judged.items() if not d["builds_ok"]]
    if not control["builds_ok"]:
        not_judged.append("control: a build failed a build check")
    not_judged += [f"{m}: no baseline loss (nothing to recover)" for m, d in judged.items()
                   if not has_baseline_loss(d["baseline"])]
    if not_judged:
        return {"verdict": "NOT JUDGED", "why": not_judged, "parts": parts}
    for m, d in judged.items():
        parts[f"{m} recovers at least 50%, interval above zero"] = recovers(d["baseline"], d["change"])
    parts["control changes by less than 2.0 points"] = control_holds(control["change"])
    return {"verdict": "PASS" if all(parts.values()) else "FAIL", "why": [], "parts": parts}
