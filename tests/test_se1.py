"""The SE1 pass rule (brokkr_edge.se1) on made-up per-image results whose right answer is known.

The numbers here are invented for the tests; they are never results.
"""

import numpy as np

from brokkr_edge import se1

N = 1000


def right(k: int) -> np.ndarray:
    """Per-image 0/1: the first k images right."""
    return (np.arange(N) < k).astype(np.float64)


FP32_CLEAN, FP32_DARK = right(800), right(780)
BASE_CLEAN, BASE_DARK = right(790), right(500)  # extra gap: (500-780) - (790-800) = -270 images


def test_baseline_loss_line_is_minus_two_points():
    e = se1.baseline_extra_gap(FP32_CLEAN, FP32_DARK, BASE_CLEAN, BASE_DARK)
    assert e["count"] == -270 and se1.has_baseline_loss(e)
    assert se1.has_baseline_loss({"count": -20, "n": 1000, "side": "below zero"})       # exactly -2.0
    assert not se1.has_baseline_loss({"count": -19, "n": 1000, "side": "below zero"})
    assert not se1.has_baseline_loss({"count": -300, "n": 1000, "side": "includes zero"})


def test_change_and_recovered_share():
    base = se1.baseline_extra_gap(FP32_CLEAN, FP32_DARK, BASE_CLEAN, BASE_DARK)
    half = se1.change(BASE_CLEAN, BASE_DARK, right(790), right(635))    # recovers 135 of 270
    less = se1.change(BASE_CLEAN, BASE_DARK, right(790), right(634))    # 134 of 270
    assert half["count"] == 135 and se1.share(base, half) == 0.5 and se1.recovers(base, half)
    assert not se1.recovers(base, less)
    assert se1.share({"count": 5}, half) is None


def test_control_line_is_strictly_below_two_points():
    assert se1.control_holds({"count": -19, "n": 1000}) and se1.control_holds({"count": 19, "n": 1000})
    assert not se1.control_holds({"count": 20, "n": 1000})
    assert not se1.control_holds({"count": -20, "n": 1000})


def test_verdict():
    base = se1.baseline_extra_gap(FP32_CLEAN, FP32_DARK, BASE_CLEAN, BASE_DARK)
    good = se1.change(BASE_CLEAN, BASE_DARK, right(790), right(700))
    flat = se1.change(BASE_CLEAN, BASE_DARK, BASE_CLEAN, BASE_DARK)
    ok = {"baseline": base, "change": good, "builds_ok": True}
    assert se1.verdict({"a": ok, "b": ok}, {"change": flat, "builds_ok": True})["verdict"] == "PASS"
    control = {"change": flat, "builds_ok": True}
    assert se1.verdict({"a": ok, "b": {**ok, "change": flat}}, control)["verdict"] == "FAIL"
    no_loss = {**ok, "baseline": {"count": -10, "n": N, "side": "below zero"}}
    v = se1.verdict({"a": ok, "b": no_loss}, {"change": flat, "builds_ok": True})
    assert v["verdict"] == "NOT JUDGED" and "no baseline loss" in v["why"][0]
    assert se1.verdict({"a": ok, "b": {**ok, "builds_ok": False}},
                       {"change": flat, "builds_ok": True})["verdict"] == "NOT JUDGED"
