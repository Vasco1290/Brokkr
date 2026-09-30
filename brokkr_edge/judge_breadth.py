"""Judge the Stage 4 breadth predictions H18-H22 (docs/hypotheses_stage4.md) from per-image results.

The rules are the ones committed before the 4.1 sweep, plus the dated note of 27 September 2026
(written before any 4.1 accuracy was computed), which fixes four details the rules left open:
- a near-floor cell (FP32 top-1 below 10% for one model at one condition) is left out by dropping that
  model from that condition's correlation (H18, H19), or from the model count (H20-H22); for H22 a
  model is left out if EITHER of its two conditions is near floor;
- a correlation needs at least 4 models; with fewer, that condition is "not judged" (the minimum
  H18-H19 were judged with; a later note, also 27 September, raised it to 6 for every correlation
  computed after them: MIN_MODELS);
- pass counts stay absolute (H18: 8 conditions, H19: 6, H20-H22: 5 models): anything left out or not
  judged can never count towards a PASS;
- Spearman correlation uses average ranks for ties (standard); in the bootstrap over models, a
  resample with fewer than 4 distinct models, or where a variable is constant (so the correlation is
  undefined), is redrawn.

Words (from the hypotheses file):
- compression-caused gap at a condition: INT8 top-1 minus FP32 top-1, same model, same images;
- extra gap: that gap minus the same model's clean gap (negative = INT8 hurts more under the damage).

Intervals: "paired" = the same resampled images for every side (brokkr_edge.judge.paired_ci: 1,000
resamples, seed 0); correlations resample the model list (1,000 resamples, seed 0). Percentile
intervals (2.5% and 97.5%), as everywhere in Brokkr.
"""

import numpy as np

from brokkr_edge.judge import N_RESAMPLES, SEED, ci_side, paired_ci

NEAR_FLOOR = 0.10  # FP32 top-1 below this under a condition: the cell is left out
MIN_MODELS = 6           # any correlation from the 27 September note on needs at least this many models
MIN_MODELS_H18_H19 = 4   # the minimum H18-H19 were judged with (fixed before they ran)
MIN_DISTINCT = 4         # a bootstrap resample with fewer distinct models than this is redrawn
LARGE_POINTS = 5   # "large" extra gap (H20, H21): at most -5.0 points


def near_floor(fp32_top1: float) -> bool:
    return fp32_top1 < NEAR_FLOOR


def average_ranks(values) -> np.ndarray:
    """Ranks 1..n; tied values share the average of the ranks they span (e.g. 1, 2.5, 2.5, 4)."""
    _, inverse, counts = np.unique(np.asarray(values, dtype=float), return_inverse=True, return_counts=True)
    last = np.cumsum(counts)  # rank of the last member of each group of equal values
    return ((last - counts + 1 + last) / 2)[inverse]


def spearman(x, y) -> float | None:
    """Spearman rank correlation (Pearson correlation of the average ranks).

    None if a variable is constant (the correlation is undefined)."""
    rx, ry = average_ranks(x), average_ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x, y) -> tuple:
    """95% interval of Spearman's correlation by resampling the models. Returns (interval, redrawn)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < MIN_DISTINCT or spearman(x, y) is None:
        raise ValueError("needs at least 4 models and a defined correlation")
    rng = np.random.default_rng(SEED)
    values, redrawn = [], 0
    while len(values) < N_RESAMPLES:
        idx = rng.integers(0, n, n)
        rho = spearman(x[idx], y[idx]) if len(np.unique(idx)) >= MIN_DISTINCT else None
        if rho is None:
            redrawn += 1
            if redrawn > 100 * N_RESAMPLES:
                raise RuntimeError("almost every resample is undefined")
            continue
        values.append(rho)
    return [float(v) for v in np.percentile(values, [2.5, 97.5])], redrawn


def correlation(models: list, x: dict, y: dict, fp32_at_condition: dict,
                min_models: int = MIN_MODELS) -> dict:
    """Spearman of x against y across models (dicts by model), near-floor models left out.

    Not judged if fewer than `min_models` remain."""
    kept = [m for m in models if not near_floor(fp32_at_condition[m])]
    found = {"models": kept, "left_out_near_floor": [m for m in models if m not in kept]}
    xs, ys = [x[m] for m in kept], [y[m] for m in kept]
    rho = spearman(xs, ys) if len(kept) >= min_models else None
    if rho is None:
        why = f"fewer than {min_models} models" if len(kept) < min_models else "a variable is constant"
        return {**found, "judged": False, "why_not_judged": why, "rho": None, "ci95": None, "side": None}
    ci, redrawn = spearman_ci(xs, ys)
    return {**found, "judged": True, "rho": rho, "ci95": ci, "side": ci_side(ci), "redrawn": redrawn}


def gap(correct_old: np.ndarray, correct_new: np.ndarray) -> dict:
    """New minus old top-1 on the same images (per-image 0/1 arrays), with its paired 95% interval."""
    per_image = correct_new - correct_old
    count = int(correct_new.sum() - correct_old.sum())
    ci = paired_ci(lambda idx: per_image[idx].mean(), len(per_image))
    return {"value": count / len(per_image), "count": count, "n": len(per_image), "ci95": ci,
            "side": ci_side(ci)}


def extra_gap(fp32_clean, int8_clean, fp32_cond, int8_cond) -> dict:
    """(INT8 - FP32 at the condition) - (INT8 - FP32 on clean images), paired over the same images."""
    per_image = (int8_cond - fp32_cond) - (int8_clean - fp32_clean)
    count = int(int8_cond.sum() - fp32_cond.sum() - int8_clean.sum() + fp32_clean.sum())
    ci = paired_ci(lambda idx: per_image[idx].mean(), len(per_image))
    return {"value": count / len(per_image), "count": count, "n": len(per_image), "ci95": ci,
            "side": ci_side(ci), "per_image": per_image}


def is_large(extra: dict) -> bool:
    """At most -5.0 points with the paired interval below zero (H20, H21).

    Compared in whole images, so no rounding can move a value across the line: count / n <= -5 / 100.
    """
    return extra["count"] * 100 <= -LARGE_POINTS * extra["n"] and extra["side"] == "below zero"


def difference_of_extra_gaps(extra_a: dict, extra_b: dict) -> dict:
    """extra_a minus extra_b for one model, paired over the same images (H22)."""
    per_image = extra_a["per_image"] - extra_b["per_image"]
    count = extra_a["count"] - extra_b["count"]
    ci = paired_ci(lambda idx: per_image[idx].mean(), len(per_image))
    return {"value": count / len(per_image), "count": count, "n": len(per_image), "ci95": ci,
            "side": ci_side(ci)}


def count_verdict(items: dict, needed: int) -> dict:
    """PASS if at least `needed` items hold. Items: name -> {"judged": bool, "holds": bool, ...}.

    Items not judged (near floor, too few models) never hold, so they can only make a PASS harder.
    """
    holding = [name for name, item in items.items() if item["judged"] and item["holds"]]
    return {"verdict": "PASS" if len(holding) >= needed else "FAIL", "needed": needed,
            "holding": len(holding), "judged": sum(item["judged"] for item in items.values()),
            "total": len(items), "items": items}


def h18_h19(models: list, conditions: list, top1: dict) -> dict:
    """H18a, H18b and H19 from top-1 values: top1[(model, precision, condition)] -> fraction correct."""
    clean_fp32 = {m: top1[(m, "fp32", "clean")] for m in models}
    clean_gap = {m: top1[(m, "int8", "clean")] - clean_fp32[m] for m in models}
    h18a, h18b, h19 = {}, {}, {}
    for c in conditions:
        fp32 = {m: top1[(m, "fp32", c)] for m in models}
        int8 = {m: top1[(m, "int8", c)] for m in models}
        cond_gap = {m: int8[m] - fp32[m] for m in models}
        a = correlation(models, clean_fp32, int8, fp32, MIN_MODELS_H18_H19)
        b = correlation(models, clean_fp32, cond_gap, fp32, MIN_MODELS_H18_H19)
        g = correlation(models, clean_gap, cond_gap, fp32, MIN_MODELS_H18_H19)
        h18a[c] = {**a, "holds": a["judged"] and a["side"] == "above zero"}
        h18b[c] = {**b, "holds": b["judged"] and b["side"] == "includes zero"}
        h19[c] = {**g, "holds": g["judged"] and g["side"] == "above zero"}
    return {"H18a": count_verdict(h18a, 8), "H18b": count_verdict(h18b, 8), "H19": count_verdict(h19, 6)}


def large_extra_gap(models: list, extras: dict, fp32_at_condition: dict) -> dict:
    """H20 / H21: at least 5 models with a large extra gap at the condition; near-floor models left out."""
    items = {}
    for m in models:
        e = {k: v for k, v in extras[m].items() if k != "per_image"}
        if near_floor(fp32_at_condition[m]):
            items[m] = {**e, "judged": False, "holds": False, "why_not_judged": "near floor"}
        else:
            items[m] = {**e, "judged": True, "holds": is_large(extras[m])}
    return count_verdict(items, 5)


def contrast_vs_noise(models: list, contrast: dict, noise: dict, fp32_contrast: dict,
                      fp32_noise: dict) -> dict:
    """H22: contrast extra gap minus noise extra gap below zero, interval below zero, in at least 5 models."""
    items = {}
    for m in models:
        d = difference_of_extra_gaps(contrast[m], noise[m])
        if near_floor(fp32_contrast[m]) or near_floor(fp32_noise[m]):
            items[m] = {**d, "judged": False, "holds": False, "why_not_judged": "near floor"}
        else:
            items[m] = {**d, "judged": True, "holds": d["count"] < 0 and d["side"] == "below zero"}
    return count_verdict(items, 5)
