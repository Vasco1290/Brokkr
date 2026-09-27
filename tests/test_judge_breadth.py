"""The H18-H22 judging rules (brokkr.judge_breadth) on made-up inputs whose right answer is known.

The numbers here are invented for the tests; they are never results.
"""

import numpy as np
import pytest

from brokkr import judge_breadth as jb

MODELS = [f"m{i}" for i in range(9)]


def test_average_ranks_share_ties():
    assert jb.average_ranks([3, 1, 2, 2]).tolist() == [4, 1, 2.5, 2.5]
    assert jb.average_ranks([5, 5, 5]).tolist() == [2, 2, 2]


def test_spearman_matches_scipy_with_ties():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = rng.integers(0, 4, 9), rng.integers(0, 4, 9)  # many ties
        if np.unique(x).size > 1 and np.unique(y).size > 1:
            assert jb.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)


def test_spearman_undefined_for_a_constant_variable():
    assert jb.spearman([1, 2, 3, 4], [7, 7, 7, 7]) is None
    assert jb.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert jb.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_ci_redraws_resamples_with_too_few_models():
    # Four models: most resamples repeat a model, so have fewer than 4 distinct ones and are redrawn.
    ci, redrawn = jb.spearman_ci([1, 2, 3, 4], [10, 20, 30, 40])
    assert ci == [1.0, 1.0] and redrawn > 0
    # Same seed, same answer.
    assert jb.spearman_ci([1, 2, 3, 4], [10, 20, 30, 40]) == (ci, redrawn)
    with pytest.raises(ValueError):
        jb.spearman_ci([1, 2, 3], [1, 2, 3])


def test_correlation_leaves_out_near_floor_models_and_needs_four():
    x = {m: i for i, m in enumerate(MODELS)}
    y = {m: 10 * i for i, m in enumerate(MODELS)}
    fp32 = {m: 0.5 for m in MODELS}
    fp32["m0"] = 0.0999  # near floor: below 10%
    fp32["m1"] = 0.10    # exactly 10% is not near floor
    r = jb.correlation(MODELS, x, y, fp32)
    assert r["judged"] and r["left_out_near_floor"] == ["m0"] and len(r["models"]) == 8
    assert r["rho"] == pytest.approx(1.0) and r["side"] == "above zero"

    fp32 = {m: (0.05 if i >= 3 else 0.5) for i, m in enumerate(MODELS)}  # only 3 models above the floor
    r = jb.correlation(MODELS, x, y, fp32)
    assert not r["judged"] and r["why_not_judged"] == "fewer than 4 models"


def paired(n: int, fp32_clean: int, int8_clean: int, fp32_cond: int, int8_cond: int) -> list:
    """Per-image 0/1 arrays with the given numbers of correct images (the first k images are right)."""
    return [(np.arange(n) < k).astype(np.float64) for k in (fp32_clean, int8_clean, fp32_cond, int8_cond)]


def test_extra_gap_counts_whole_images():
    # Clean: INT8 loses 10 of 1,000. Condition: INT8 loses 80. Extra gap = -70 images = -7.0 points.
    e = jb.extra_gap(*paired(1000, 800, 790, 600, 520))
    assert e["count"] == -70 and e["value"] == pytest.approx(-0.07) and e["n"] == 1000
    assert e["side"] == "below zero" and e["ci95"][1] < 0


def test_is_large_boundary_is_exactly_minus_five_points():
    below = {"side": "below zero"}
    assert jb.is_large({**below, "count": -500, "n": 10_000})      # exactly -5.00 points: "at most -5.0"
    assert not jb.is_large({**below, "count": -499, "n": 10_000})  # -4.99 points
    assert not jb.is_large({"side": "includes zero", "count": -900, "n": 10_000})


def test_count_verdict_never_counts_items_left_out():
    items = {f"x{i}": {"judged": True, "holds": True} for i in range(4)}
    items["floor"] = {"judged": False, "holds": True}  # even if marked, a left-out item never counts
    v = jb.count_verdict(items, 5)
    assert v["verdict"] == "FAIL" and v["holding"] == 4 and v["judged"] == 4 and v["total"] == 5
    items["x4"] = {"judged": True, "holds": True}
    assert jb.count_verdict(items, 5)["verdict"] == "PASS"


def test_large_extra_gap_and_contrast_vs_noise():
    big = {m: jb.extra_gap(*paired(1000, 800, 790, 600, 520)) for m in MODELS}    # -7.0 points
    small = {m: jb.extra_gap(*paired(1000, 800, 790, 600, 590)) for m in MODELS}  # 0 points
    fp32 = {m: 0.6 for m in MODELS}
    assert jb.large_extra_gap(MODELS, big, fp32)["verdict"] == "PASS"
    assert jb.large_extra_gap(MODELS, small, fp32)["verdict"] == "FAIL"

    # Five models large, but one of them is near floor: only 4 count, so FAIL (absolute count of 5).
    mixed = {m: (big[m] if i < 5 else small[m]) for i, m in enumerate(MODELS)}
    floor = {**fp32, "m0": 0.05}
    v = jb.large_extra_gap(MODELS, mixed, floor)
    assert v["verdict"] == "FAIL" and v["holding"] == 4 and not v["items"]["m0"]["judged"]

    v = jb.contrast_vs_noise(MODELS, big, small, fp32, fp32)
    assert v["verdict"] == "PASS" and v["items"]["m0"]["count"] == -70
    # H22 leaves a model out if EITHER condition is near floor.
    v = jb.contrast_vs_noise(MODELS, big, small, fp32, {**fp32, "m0": 0.01})
    assert not v["items"]["m0"]["judged"] and v["holding"] == 8
    assert jb.contrast_vs_noise(MODELS, small, big, fp32, fp32)["verdict"] == "FAIL"


def test_h18_h19_on_a_made_up_table():
    # Clean FP32 top-1 rises with the model index; INT8 under damage too (H18a holds everywhere),
    # and INT8's gap shrinks with the index, the same order on clean and damaged images (H19 holds).
    conditions = [f"c{j}" for j in range(12)]
    top1 = {}
    for i, m in enumerate(MODELS):
        top1[(m, "fp32", "clean")] = 0.60 + 0.02 * i
        top1[(m, "int8", "clean")] = top1[(m, "fp32", "clean")] - 0.01 * (9 - i)
        for c in conditions:
            top1[(m, "fp32", c)] = 0.40 + 0.03 * i
            top1[(m, "int8", c)] = top1[(m, "fp32", c)] - 0.02 * (9 - i)
    v = jb.h18_h19(MODELS, conditions, top1)
    assert v["H18a"]["verdict"] == "PASS" and v["H18a"]["holding"] == 12
    assert v["H19"]["verdict"] == "PASS"
    # Here the gap is perfectly ranked by clean accuracy, so the H18b interval excludes zero: FAIL.
    assert v["H18b"]["verdict"] == "FAIL" and v["H18b"]["holding"] == 0
