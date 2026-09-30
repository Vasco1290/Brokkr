"""Selective prediction: how much does the error rate drop if the model skips its least-sure images?

    Sort images from most to least confident. "Coverage" = the share of images the model
    answers (the most confident ones); "risk" = the error rate among those answered images.
    A model that knows when it's wrong puts its mistakes at the unconfident end, so its risk
    falls steeply as coverage shrinks.

    AURC = area under the risk-coverage curve (average risk over all coverage levels);
           lower is better. (Geifman & El-Yaniv, 2017; Geifman et al., 2019)
    E-AURC = AURC minus the best AURC possible with the same number of mistakes (mistakes all
           ranked last). It separates "how often is it wrong" from "does its confidence rank its
           mistakes", so models with different accuracy can be compared fairly.

Ties: if several images have exactly the same confidence (common for INT8, whose scores are
rounded), their order is arbitrary. Instead of picking one order, the curve uses the average
over all orders: inside a tied group, errors are spread evenly.

Uses only NumPy, so it can be shared with the Argos project.
"""

import numpy as np

COVERAGE_GRID = np.round(np.linspace(0.01, 1.0, 100), 2)  # 1%, 2%, ... 100%


def risk_curve(confidence: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """risk[k-1] = expected error rate among the k most confident images, for k = 1..n."""
    order = np.argsort(-confidence, kind="stable")
    conf, wrong = confidence[order], 1.0 - correct[order]
    n = len(conf)
    # Label each image with its group of equal confidence (0, 0, 1, 2, 2, 2, ...).
    group = np.cumsum(np.r_[True, conf[1:] != conf[:-1]]) - 1
    group_size = np.bincount(group)
    group_errors = np.bincount(group, weights=wrong)
    group_start = np.r_[0, np.cumsum(group_size)[:-1]]
    errors_before_group = np.r_[0.0, np.cumsum(group_errors)[:-1]]
    # After taking j images of a group (in random order), expect j * that group's error rate.
    j = np.arange(n) - group_start[group] + 1
    expected_errors = errors_before_group[group] + j * group_errors[group] / group_size[group]
    return expected_errors / np.arange(1, n + 1)


def aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    return float(risk_curve(confidence, correct).mean())


def optimal_aurc(correct: np.ndarray) -> float:
    """AURC if every mistake were ranked least confident (the best any ranking can do)."""
    n, n_correct = len(correct), int(correct.sum())
    k = np.arange(1, n + 1)
    return float((np.maximum(0, k - n_correct) / k).mean())


def selective_prediction(confidence: np.ndarray, correct: np.ndarray,
                         n_resamples: int = 1000, seed: int = 0) -> dict:
    """AURC, E-AURC (with bootstrap 95% CIs), and the risk-coverage curve on a 1% grid."""
    curve = risk_curve(confidence, correct)
    n = len(confidence)
    rng = np.random.default_rng(seed)
    boot_aurc, boot_eaurc = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        a = aurc(confidence[idx], correct[idx])
        boot_aurc.append(a)
        boot_eaurc.append(a - optimal_aurc(correct[idx]))

    def risk_at(coverage: float) -> float:
        return float(curve[max(1, int(round(coverage * n))) - 1])

    a, opt = float(curve.mean()), optimal_aurc(correct)
    return {
        "aurc": a,
        "aurc_ci95": [float(x) for x in np.percentile(boot_aurc, [2.5, 97.5])],
        "e_aurc": a - opt,
        "e_aurc_ci95": [float(x) for x in np.percentile(boot_eaurc, [2.5, 97.5])],
        "optimal_aurc": opt,
        "risk_at_full_coverage": risk_at(1.0),
        "risk_at_80pct_coverage": risk_at(0.8),
        "risk_at_50pct_coverage": risk_at(0.5),
        "curve": {"coverage": COVERAGE_GRID.tolist(), "risk": [risk_at(c) for c in COVERAGE_GRID]},
    }
