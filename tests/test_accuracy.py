"""Checks for brokkr.accuracy (no real dataset needed)."""

import numpy as np
import pytest
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights

from brokkr.accuracy import bootstrap_ci, preprocess, topk_correct, unpaired_bootstrap_diff


def test_preprocess_matches_torchvision():
    # A random non-square picture, processed by our code and by torchvision's own transforms.
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))

    ours = preprocess(image)
    theirs = MobileNet_V3_Large_Weights.IMAGENET1K_V2.transforms()(image).numpy()

    assert ours.shape == theirs.shape == (3, 224, 224)
    assert np.abs(ours - theirs).max() < 1e-4


def test_preprocess_handles_greyscale():
    assert preprocess(Image.new("L", (250, 250))).shape == (3, 224, 224)


def test_topk_correct():
    logits = np.array([[0.1, 0.9, 0.0],   # predicts class 1, then 0
                       [0.8, 0.1, 0.2]])  # predicts class 0, then 2
    labels = np.array([0, 2])
    assert topk_correct(logits, labels, k=1).tolist() == [0, 0]
    assert topk_correct(logits, labels, k=2).tolist() == [1, 1]


def test_bootstrap_ci_contains_accuracy_and_shrinks_with_more_images():
    rng = np.random.default_rng(0)
    small = (rng.random(100) < 0.75).astype(float)
    large = (rng.random(10_000) < 0.75).astype(float)

    lo_s, hi_s = bootstrap_ci(small)
    lo_l, hi_l = bootstrap_ci(large)
    assert lo_s <= small.mean() <= hi_s
    assert lo_l <= large.mean() <= hi_l
    assert (hi_l - lo_l) < (hi_s - lo_s)  # more images -> more certain


def test_bootstrap_is_reproducible():
    data = (np.random.default_rng(1).random(500) < 0.5).astype(float)
    assert bootstrap_ci(data, seed=3) == bootstrap_ci(data, seed=3)


def test_accuracy_from_logits_on_known_scores():
    from brokkr.accuracy import accuracy_from_logits
    logits = np.array([[0.1, 3.0, 0.2, 0.0, -1.0, -2.0],   # top-1 is class 1
                       [2.0, 0.0, 0.1, 0.2, 0.3, 0.4]])    # top-1 is class 0, class 1 is last
    result = accuracy_from_logits(logits, np.array([1, 1]))
    assert result["metrics"]["top1"] == 0.5
    assert result["metrics"]["top5"] == 0.5  # class 1 is 6th of 6 for the second image
    assert result["raw"]["top5_predictions"][0][0] == 1


def test_ties_are_broken_consistently_and_reported():
    from brokkr.accuracy import accuracy_from_logits
    from brokkr.shift.reliability import confidence_and_correct
    logits = np.array([[5.0, 5.0, 1.0, 0.0, 0.0, 0.0],   # classes 0 and 1 tie; label 1
                       [5.0, 5.0, 1.0, 0.0, 0.0, 0.0],   # same tie; label 0
                       [0.0, 9.0, 1.0, 0.0, 0.0, 0.0]])  # no tie; label 1
    labels = np.array([1, 0, 1])
    m = accuracy_from_logits(logits, labels)["metrics"]
    assert m["top1"] == pytest.approx(2 / 3)  # lower class wins the tie: image 2 right, image 1 wrong
    assert m["top1_tied_images"] == 2
    assert m["top1_range_over_tie_breaks"] == pytest.approx([1 / 3, 1.0])
    # Calibration code must agree with the accuracy code on which answer the model gave.
    _, correct = confidence_and_correct(logits, labels)
    assert correct.mean() == pytest.approx(m["top1"])


def test_unpaired_diff_matches_the_textbook_interval():
    rng = np.random.default_rng(0)
    a = (rng.random(10_000) < 0.75).astype(float)
    b = (rng.random(5_000) < 0.74).astype(float)
    diff, low, high = unpaired_bootstrap_diff(a, b)
    assert diff == pytest.approx(b.mean() - a.mean())
    assert low < diff < high
    # Normal approximation for two independent proportions: half-width 1.96 * sqrt(pa(1-pa)/na + ...)
    half = 1.96 * np.sqrt(a.mean() * (1 - a.mean()) / len(a) + b.mean() * (1 - b.mean()) / len(b))
    assert (high - low) / 2 == pytest.approx(half, rel=0.15)
    assert unpaired_bootstrap_diff(a, b) == (diff, low, high)  # same seed, same answer
