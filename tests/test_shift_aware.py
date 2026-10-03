"""Checks for the shift-aware tools: the confidence alarm and the robust conformal mix."""

from collections import Counter

import numpy as np
import pytest

from brokkr_edge.shift.alarm import alarm_threshold, consecutive_window_means, fires, random_window_means
from brokkr_edge.shift.robust_conformal import robust_calibration_plan


def test_alarm_fires_on_about_one_percent_of_clean_windows():
    confidence = np.random.default_rng(0).uniform(0.3, 1.0, 5_000)
    threshold = alarm_threshold(confidence)
    share = fires(random_window_means(confidence), threshold).mean()
    assert share == pytest.approx(0.01, abs=0.001)  # the 1st percentile, by construction


def test_alarm_fires_when_confidence_drops():
    rng = np.random.default_rng(0)
    clean, damaged = rng.uniform(0.5, 1.0, 5_000), rng.uniform(0.3, 0.8, 10_000)
    threshold = alarm_threshold(clean)
    assert fires(consecutive_window_means(damaged), threshold).all()


def test_consecutive_windows_do_not_overlap_and_use_a_fixed_order():
    confidence = np.arange(10_000, dtype=float)
    means = consecutive_window_means(confidence)
    assert len(means) == 100
    assert means.mean() == pytest.approx(confidence.mean())  # every image used exactly once
    assert np.array_equal(means, consecutive_window_means(confidence))  # same seed, same windows


def test_random_windows_hold_different_images():
    rng_check = np.random.default_rng(4)
    first = rng_check.choice(5_000, 100, replace=False)
    confidence = np.arange(5_000, dtype=float)
    assert random_window_means(confidence, n_windows=1)[0] == pytest.approx(confidence[first].mean())
    assert len(set(first)) == 100


def test_robust_plan_is_one_third_clean_and_balanced():
    allowed = ["fog", "defocus_blur", "motion_blur", "noise"]
    plan = robust_calibration_plan(5_000, allowed)
    counts = Counter(plan)
    assert counts[(None, 0)] == 1_667
    damaged = {pair: n for pair, n in counts.items() if pair[0] is not None}
    assert len(damaged) == 20 and set(damaged.values()) <= {166, 167}
    assert {name for name, _ in damaged} == set(allowed)


def test_robust_plan_uses_the_same_images_for_every_held_out_corruption():
    a = robust_calibration_plan(5_000, ["fog", "defocus_blur", "motion_blur", "noise"])
    b = robust_calibration_plan(5_000, ["defocus_blur", "motion_blur", "noise", "darkness"])
    assert [x[0] is None for x in a] == [x[0] is None for x in b]  # same clean images
    assert [x[1] for x in a] == [x[1] for x in b]  # same severities
