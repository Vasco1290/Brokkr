"""A "conditions changed" alarm: watch the average confidence of the last N images.

Stage 2 showed that when images get damaged, prediction sets barely grow but the model's
confidence (probability of its top answer) does fall. So a simple alarm: average the confidence
over a window of 100 images and raise a warning when that average drops below a threshold.

The threshold comes from clean images: draw many random windows of clean images, and take the
average confidence that only 1% of clean windows fall below. On clean images the alarm then fires
about 1% of the time (a false alarm); on damaged images it should fire much more often.

Uses only NumPy, so it can be shared with the Argos project.
"""

import numpy as np


def random_window_means(confidence: np.ndarray, window: int = 100, n_windows: int = 10_000,
                        seed: int = 4) -> np.ndarray:
    """Average confidence of `n_windows` random windows, each of `window` different images.

    Windows are drawn one after another from one random generator; two windows may share images.
    """
    rng = np.random.default_rng(seed)
    return np.array([confidence[rng.choice(len(confidence), window, replace=False)].mean()
                     for _ in range(n_windows)])


def alarm_threshold(confidence: np.ndarray, window: int = 100, n_windows: int = 10_000,
                    percentile: float = 1.0, seed: int = 4) -> float:
    """The window average that only `percentile`% of random clean windows fall below."""
    return float(np.percentile(random_window_means(confidence, window, n_windows, seed), percentile))


def consecutive_window_means(confidence: np.ndarray, window: int = 100, seed: int = 3) -> np.ndarray:
    """Put the images in one fixed random order and average each block of `window` consecutive images.

    Blocks do not overlap. The same seed gives the same order, so every condition and every model
    is cut into windows of the same images.
    """
    order = np.random.default_rng(seed).permutation(len(confidence))
    n = len(confidence) // window
    return confidence[order[:n * window]].reshape(n, window).mean(axis=1)


def fires(window_means: np.ndarray, threshold: float) -> np.ndarray:
    """True for each window whose average confidence is strictly below the threshold."""
    return window_means < threshold
