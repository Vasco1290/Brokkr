"""Damage seed schemes (brokkr_edge.seeds; docs/user_models.md, D18). Made-up pictures only."""

import hashlib

import numpy as np

from brokkr_edge import sweep
from brokkr_edge.seeds import content_seed, damaged_batch_seeded

SHA = "ab" * 32  # a made-up image fingerprint


def test_content_seed_follows_the_written_rule():
    expected = int.from_bytes(hashlib.sha256(f"{SHA}:brokkr/fog".encode()).digest()[:4], "big")
    assert content_seed(SHA, "brokkr", "fog") == expected
    assert content_seed(SHA.upper(), "brokkr", "fog") == expected  # hex case does not matter
    assert 0 <= expected < 2**32


def test_content_seed_differs_by_image_suite_and_damage_type():
    seeds = {content_seed(SHA, "brokkr", "fog"), content_seed("cd" * 32, "brokkr", "fog"),
             content_seed(SHA, "imagenet-c", "fog"), content_seed(SHA, "brokkr", "noise")}
    assert len(seeds) == 4


def test_with_position_seeds_the_seeded_batch_equals_the_studys_batch():
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 256, (6, 32, 32, 3), dtype=np.uint8)
    positions = np.array([3, 17, 40, 41, 99, 1234])
    indices = [0, 2, 5]
    conditions = (("brokkr", "fog", 3), ("brokkr", "noise", 3), ("brokkr", "darkness", 5),
                  ("imagenet-c", "gaussian_noise", 5), ("imagenet-c", "fog", 3))
    for suite, corruption, severity in conditions:
        study = sweep.damaged_batch(crops, positions, indices, suite, corruption, severity)
        seeded = damaged_batch_seeded(crops, indices, suite, corruption, severity,
                                      [sweep.seed_for(positions[i]) for i in indices])
        assert np.array_equal(study, seeded), (suite, corruption)


def test_other_seeds_change_random_damage_only():
    crops = np.random.default_rng(1).integers(0, 256, (2, 32, 32, 3), dtype=np.uint8)
    a = damaged_batch_seeded(crops, [0, 1], "brokkr", "noise", 3, [1, 2])
    b = damaged_batch_seeded(crops, [0, 1], "brokkr", "noise", 3, [3, 4])
    assert not np.array_equal(a, b)
    dark_a = damaged_batch_seeded(crops, [0, 1], "brokkr", "darkness", 5, [1, 2])
    dark_b = damaged_batch_seeded(crops, [0, 1], "brokkr", "darkness", 5, [3, 4])
    assert np.array_equal(dark_a, dark_b)  # darkness draws no random numbers
