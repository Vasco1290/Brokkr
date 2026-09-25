"""Checks for brokkr.shift.reliability, using simulated models whose true calibration we know."""

import numpy as np
import pytest

from brokkr.shift.reliability import calibration, ece, reliability_bins, softmax


def test_softmax_sums_to_one_and_survives_huge_scores():
    probs = softmax(np.array([[1000.0, 999.0, -1000.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(np.isfinite(probs))
    assert probs[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_honest_model_has_ece_near_zero():
    # Simulate a model that is right with exactly the probability it claims.
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.1, 1.0, 200_000)
    correct = (rng.random(200_000) < confidence).astype(float)
    assert ece(confidence, correct) < 0.01


def test_overconfident_model_has_the_expected_ece():
    # Claims 90% every time, is right 60% of the time -> 30 points off.
    rng = np.random.default_rng(1)
    confidence = np.full(100_000, 0.9)
    correct = (rng.random(100_000) < 0.6).astype(float)
    assert ece(confidence, correct) == pytest.approx(0.30, abs=0.01)


def test_bins_cover_every_image_once():
    rng = np.random.default_rng(2)
    confidence = rng.uniform(0, 1, 1000)
    confidence[:3] = [0.0, 1.0, 1 / 15]  # edges must land in a bin too
    correct = (rng.random(1000) < 0.5).astype(float)
    bins = reliability_bins(confidence, correct)
    assert sum(b["count"] for b in bins) == 1000
    assert len(bins) == 15


def test_calibration_from_logits():
    # Two classes; logits chosen so the model says 88% for class 0 on every image.
    logits = np.tile([2.0, 0.0], (1000, 1))
    labels = np.array([0] * 700 + [1] * 300)  # right 70% of the time
    result = calibration(logits, labels, n_resamples=200)
    assert result["mean_confidence"] == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert result["accuracy"] == pytest.approx(0.7)
    assert result["overconfidence"] == pytest.approx(result["mean_confidence"] - 0.7)
    assert result["ece"] == pytest.approx(result["overconfidence"])  # one bin, so ECE = the gap
    lo, hi = result["ece_ci95"]
    assert lo <= result["ece"] <= hi


def test_ece_is_biased_upwards_for_an_honest_model():
    # Why the ECE interval can miss the measured value: for a nearly perfectly calibrated model,
    # resampled ECEs mostly come out HIGHER than the measured one, because ECE can't go below 0.
    rng = np.random.default_rng(5)
    confidence = rng.uniform(0.1, 1.0, 2000)
    correct = (rng.random(2000) < confidence).astype(float)
    measured = ece(confidence, correct)
    resampled = [ece(confidence[i], correct[i]) for i in (rng.integers(0, 2000, 2000) for _ in range(300))]
    assert np.mean(np.array(resampled) > measured) > 0.7
