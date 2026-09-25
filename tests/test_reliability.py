"""Checks for brokkr.shift.reliability, using simulated models whose true calibration we know."""

import numpy as np
import pytest

from brokkr.shift.reliability import calibration, ece, fit_temperature, nll, reliability_bins, softmax


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


def test_nll_matches_the_definition():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    labels = np.array([0, 2])
    expected = -np.mean(np.log(softmax(logits / 2.0)[[0, 1], labels]))
    assert nll(logits, labels, temperature=2.0) == pytest.approx(expected)


@pytest.mark.parametrize("true_temperature", [0.5, 2.0])
def test_fit_temperature_recovers_a_known_temperature(true_temperature):
    # Labels drawn from softmax(scores / T): the best-fitting temperature is close to T.
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(20_000, 10))
    probs = softmax(logits / true_temperature)
    labels = np.array([rng.choice(10, p=p) for p in probs])
    t = fit_temperature(logits, labels)
    assert t == pytest.approx(true_temperature, rel=0.05)
    assert nll(logits, labels, t) <= min(nll(logits, labels, t * 1.01), nll(logits, labels, t / 1.01))


def test_fit_temperature_stays_inside_its_range():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 1, size=(2_000, 10))
    labels = logits.argmax(axis=1)  # always right: the best T is as small as allowed
    assert fit_temperature(logits, labels, low=0.5, high=2.0) == pytest.approx(0.5, rel=1e-3)
