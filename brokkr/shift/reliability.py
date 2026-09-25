"""Does a classifier know when it's wrong? Reliability measurements from its raw scores.

Calibration: when the model says "80% sure", is it right about 80% of the time?

    Confidence = the probability the model gives its top answer.
    Images are grouped into bins by confidence (0-6.7%, 6.7-13.3%, ... 93.3-100%).
    In each bin we compare the average confidence with the actual accuracy.
    ECE (expected calibration error) = the average gap, weighted by how many images fall in
    each bin. 0 = perfectly honest confidence. 0.05 = on average 5 percentage points off.
    Overconfidence = average confidence minus accuracy: positive means it claims more than it
    delivers.

Standard setup from Guo et al., "On Calibration of Modern Neural Networks" (2017): 15 equal-width
bins. Uses only NumPy, so it can be shared with the Argos project.

Caution: ECE can't go below 0, so random noise in a finite test set can only push it up. It is
biased upwards, most of all when the model is nearly perfectly calibrated. The bootstrap interval
inherits this: for very small ECE (below about 0.02) the whole interval can sit above the measured
value. Don't over-read small ECE differences.
"""

import numpy as np

N_BINS = 15


def softmax(logits: np.ndarray) -> np.ndarray:
    """Turn raw scores into probabilities that sum to 1 per image (numerically stable)."""
    shifted = logits - logits.max(axis=1, keepdims=True)  # avoids overflow in exp
    exp = np.exp(shifted.astype(np.float64))
    return exp / exp.sum(axis=1, keepdims=True)


def confidence_and_correct(logits: np.ndarray, labels: np.ndarray) -> tuple:
    """Per image: the probability of the top answer, and 1.0 if that answer is right."""
    probs = softmax(logits)
    predicted = probs.argmax(axis=1)
    return probs.max(axis=1), (predicted == labels).astype(np.float64)


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error from per-image confidence and correctness."""
    # Bin index: confidence in (0, 1/15] -> 0, ..., (14/15, 1] -> 14.
    bins = np.clip(np.ceil(confidence * n_bins).astype(int) - 1, 0, n_bins - 1)
    conf_sum = np.bincount(bins, weights=confidence, minlength=n_bins)
    correct_sum = np.bincount(bins, weights=correct, minlength=n_bins)
    # sum over bins of (bin size / total) * |mean confidence - accuracy| simplifies to
    # sum over bins of |confidence sum - correct count| / total.
    return float(np.abs(conf_sum - correct_sum).sum() / len(confidence))


def reliability_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> list:
    """Per bin: its range, how many images, average confidence, and actual accuracy.

    This is the data behind a reliability diagram (confidence vs accuracy).
    """
    bins = np.clip(np.ceil(confidence * n_bins).astype(int) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        in_bin = bins == b
        n = int(in_bin.sum())
        rows.append({
            "low": b / n_bins, "high": (b + 1) / n_bins, "count": n,
            "mean_confidence": float(confidence[in_bin].mean()) if n else None,
            "accuracy": float(correct[in_bin].mean()) if n else None,
        })
    return rows


def calibration(logits: np.ndarray, labels: np.ndarray, n_resamples: int = 1000, seed: int = 0) -> dict:
    """All calibration numbers for one set of predictions, with a bootstrap 95% CI for ECE."""
    confidence, correct = confidence_and_correct(logits, labels)
    rng = np.random.default_rng(seed)
    n = len(confidence)
    resampled = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        resampled.append(ece(confidence[idx], correct[idx]))
    low, high = np.percentile(resampled, [2.5, 97.5])
    return {
        "ece": ece(confidence, correct),
        "ece_ci95": [float(low), float(high)],
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
        "overconfidence": float(confidence.mean() - correct.mean()),
        "n_bins": N_BINS,
        "bins": reliability_bins(confidence, correct),
    }
