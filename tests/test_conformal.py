"""Checks for brokkr_edge.shift.conformal, on simulated data where the maths guarantees the answer."""

import numpy as np
import pytest

from brokkr_edge.shift.conformal import conformal_threshold, evaluate_sets, prediction_sets
from brokkr_edge.shift.reliability import softmax


def simulated_model(n: int, n_classes: int, rng: np.random.Generator) -> tuple:
    """Random 'model' probabilities and labels drawn from them (images are exchangeable)."""
    probs = softmax(rng.standard_normal((n, n_classes)) * 2.0)
    labels = np.array([rng.choice(n_classes, p=p) for p in probs])
    return probs, labels


def test_threshold_on_a_small_known_example():
    # Probability of the right class: 0.9, 0.8, 0.6, 0.3 -> scores 0.1, 0.2, 0.4, 0.7.
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    labels = np.array([0, 1, 0, 1])
    # coverage 0.5: level ceil(5 * 0.5) / 4 = 0.75 -> 3rd smallest score = 0.4
    assert conformal_threshold(probs, labels, coverage=0.5) == pytest.approx(0.4)


def test_coverage_lands_at_the_target_on_exchangeable_data():
    # Averaged over many random calibration/test splits, coverage must be at least 90%
    # and not much more (the guarantee is at most 90% + 1/(n+1) on average).
    rng = np.random.default_rng(0)
    coverages = []
    for _ in range(40):
        probs, labels = simulated_model(3000, 20, rng)
        q = conformal_threshold(probs[:1000], labels[:1000], coverage=0.9)
        coverages.append(evaluate_sets(probs[1000:], labels[1000:], q, n_resamples=10)["coverage"])
    assert np.mean(coverages) == pytest.approx(0.90, abs=0.01)
    assert np.mean(coverages) >= 0.895


def test_higher_target_gives_bigger_sets():
    rng = np.random.default_rng(1)
    probs, labels = simulated_model(4000, 20, rng)
    sizes = [prediction_sets(probs[2000:], conformal_threshold(probs[:2000], labels[:2000], c)).sum(1).mean()
             for c in (0.8, 0.9, 0.95)]
    assert sizes[0] < sizes[1] < sizes[2]


def test_sets_are_confidence_ordered():
    # If a class is in the set, every class the model finds more likely is in it too.
    rng = np.random.default_rng(2)
    probs, labels = simulated_model(500, 10, rng)
    sets = prediction_sets(probs, conformal_threshold(probs, labels, 0.9))
    for p, s in zip(probs, sets, strict=True):
        if s.any():
            assert p[s].min() >= p[~s].max(initial=0.0)


def test_report_fields_are_consistent():
    rng = np.random.default_rng(3)
    probs, labels = simulated_model(2000, 10, rng)
    q = conformal_threshold(probs[:1000], labels[:1000])
    result = evaluate_sets(probs[1000:], labels[1000:], q, n_resamples=100)
    assert sum(result["set_size_counts"].values()) == 1000
    lo, hi = result["coverage_ci95"]
    assert lo <= result["coverage"] <= hi
    assert 0 <= result["share_empty"] <= 1 and 0 <= result["share_single_class"] <= 1


def test_too_few_calibration_images_gives_every_class():
    # With 5 images, a 90% promise needs the 6th smallest score, which doesn't exist.
    probs = np.full((5, 3), 1 / 3)
    assert conformal_threshold(probs, np.zeros(5, dtype=int), coverage=0.9) == 1.0
    assert prediction_sets(probs, 1.0).all()
