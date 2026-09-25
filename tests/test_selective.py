"""Checks for brokkr.shift.selective, on simulated models whose behaviour we know."""

import numpy as np
import pytest

from brokkr.shift.selective import aurc, optimal_aurc, risk_curve, selective_prediction


def test_small_example_by_hand():
    # Confidence order: 0.9 (right), 0.8 (wrong), 0.7 (right), 0.6 (wrong)
    confidence = np.array([0.7, 0.9, 0.6, 0.8])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    # errors among top k: 0, 1, 1, 2 -> risk 0/1, 1/2, 1/3, 2/4
    assert risk_curve(confidence, correct) == pytest.approx([0, 1 / 2, 1 / 3, 2 / 4])
    assert aurc(confidence, correct) == pytest.approx((0 + 1 / 2 + 1 / 3 + 1 / 2) / 4)


def test_perfect_ranking_reaches_the_optimum():
    rng = np.random.default_rng(0)
    correct = (rng.random(5000) < 0.7).astype(float)
    confidence = correct + 0.1 * rng.random(5000)  # every right answer more confident than any wrong one
    result = selective_prediction(confidence, correct, n_resamples=50)
    assert result["aurc"] == pytest.approx(optimal_aurc(correct))
    assert result["e_aurc"] == pytest.approx(0.0, abs=1e-12)
    assert result["risk_at_50pct_coverage"] == 0.0


def test_random_confidence_gains_nothing_from_skipping():
    rng = np.random.default_rng(1)
    correct = (rng.random(20_000) < 0.6).astype(float)
    result = selective_prediction(rng.random(20_000), correct, n_resamples=50)
    error = 1 - correct.mean()
    assert result["risk_at_50pct_coverage"] == pytest.approx(error, abs=0.02)
    assert result["aurc"] == pytest.approx(error, abs=0.02)


def test_all_tied_confidence_gives_a_flat_curve_exactly():
    # Every image equally confident: any order is arbitrary, so the expected curve is flat.
    correct = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    curve = risk_curve(np.full(5, 0.5), correct)
    assert curve == pytest.approx(np.full(5, 0.4))


def test_ties_do_not_depend_on_input_order():
    rng = np.random.default_rng(2)
    confidence = rng.integers(0, 5, 1000) / 5.0  # only 5 distinct values: lots of ties
    correct = (rng.random(1000) < 0.5).astype(float)
    shuffle = rng.permutation(1000)
    assert aurc(confidence, correct) == pytest.approx(aurc(confidence[shuffle], correct[shuffle]))


def test_worse_ranking_gives_higher_aurc():
    rng = np.random.default_rng(3)
    correct = (rng.random(5000) < 0.7).astype(float)
    good = correct + 0.5 * rng.random(5000)
    bad = (1 - correct) + 0.5 * rng.random(5000)  # confident when wrong
    assert aurc(good, correct) < aurc(rng.random(5000), correct) < aurc(bad, correct)


def test_report_fields():
    rng = np.random.default_rng(4)
    correct = (rng.random(2000) < 0.7).astype(float)
    result = selective_prediction(rng.random(2000) + correct, correct, n_resamples=100)
    assert len(result["curve"]["coverage"]) == len(result["curve"]["risk"]) == 100
    assert result["curve"]["risk"][-1] == pytest.approx(1 - correct.mean())
    lo, hi = result["aurc_ci95"]
    assert lo <= result["aurc"] <= hi
