"""Checks for brokkr.accuracy (no real dataset needed)."""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights

from brokkr.accuracy import bootstrap_ci, preprocess, topk_correct


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
