"""Conformal prediction: answer with a SET of classes that contains the right one 90% of the time.

How it works (the standard "LAC" method, Sadinle et al. 2019; see Angelopoulos & Bates,
"A Gentle Introduction to Conformal Prediction", 2021):

1. Calibrate. On images the model has never been tuned on, look at the probability the model
   gave the RIGHT class. Its "nonconformity score" is 1 - that probability (high = the model
   was surprised by the truth).
2. Pick the threshold q so that about 90% of calibration images have a score <= q
   (a slightly conservative quantile, so the promise holds exactly for finite samples).
3. Predict. For a new image, the set is every class whose probability is at least 1 - q.
   Confident image -> small set (often one class). Unsure image -> big set.

The promise: if new images come from the same distribution as the calibration images, the set
contains the right class at least 90% of the time. If they don't (fog, blur, ...), there is no
promise, and measuring how far coverage falls is the point of the stress test.

Uses only NumPy, so it can be shared with the Argos project.
"""

import numpy as np


def conformal_threshold(cal_probs: np.ndarray, cal_labels: np.ndarray, coverage: float = 0.9) -> float:
    """Threshold q from calibration data, for sets that contain the right class `coverage` of the time."""
    n = len(cal_labels)
    scores = np.sort(1.0 - cal_probs[np.arange(n), cal_labels])
    # The k-th smallest score, k = ceil((n + 1) * coverage), gives the finite-sample guarantee.
    # (Written out exactly: np.quantile(..., method="higher") can land one rank too high.)
    k = int(np.ceil((n + 1) * coverage))
    if k > n:
        return 1.0  # too few calibration images for this promise: every class goes in the set
    return float(scores[k - 1])


def prediction_sets(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask (n_images, n_classes): True where a class is in that image's set."""
    return probs >= 1.0 - threshold


def evaluate_sets(probs: np.ndarray, labels: np.ndarray, threshold: float,
                  n_resamples: int = 1000, seed: int = 0) -> dict:
    """Coverage and set size on test images, with bootstrap 95% intervals."""
    sets = prediction_sets(probs, threshold)
    covered = sets[np.arange(len(labels)), labels].astype(np.float64)
    sizes = sets.sum(axis=1).astype(np.float64)

    rng = np.random.default_rng(seed)
    n = len(labels)
    boot_cov, boot_size = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        boot_cov.append(covered[idx].mean())
        boot_size.append(sizes[idx].mean())

    return {
        "coverage": float(covered.mean()),
        "coverage_ci95": [float(x) for x in np.percentile(boot_cov, [2.5, 97.5])],
        "mean_set_size": float(sizes.mean()),
        "mean_set_size_ci95": [float(x) for x in np.percentile(boot_size, [2.5, 97.5])],
        "median_set_size": float(np.median(sizes)),
        "share_single_class": float((sizes == 1).mean()),  # the model commits to one answer
        "share_empty": float((sizes == 0).mean()),  # possible with this method: "none look likely"
        "set_size_counts": {str(k): int(v) for k, v in zip(*np.unique(sizes.astype(int), return_counts=True),
                                                             strict=True)},
    }
