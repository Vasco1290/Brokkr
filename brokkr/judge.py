"""Judge the Stage 3 predictions H10-H17 (docs/hypotheses_stage3.md) from saved model scores.

Every function takes raw scores (logits) and labels, computes the numbers its prediction talks
about, and applies ONLY the thresholds written in the hypotheses file before the final run.
Each returns a dict:

    {"verdict": "PASS" | "FAIL" | "NOT RUN",   PASS = every part holds
     "parts": {description: True/False},       the individual conditions
     "numbers": {...},                         what was measured, with paired 95% intervals
     "within_noise": True/False/None,          INT8-vs-INT8 difference smaller than build luck
     "notes": [...]}                           reporting rules (e.g. H11 under 15%)

"Paired CI" = paired bootstrap 95% interval of a difference: the same resampled test images for both
sides, 1,000 resamples, seed 0.
"""

import numpy as np

from brokkr.shift.alarm import consecutive_window_means, fires
from brokkr.shift.conformal import evaluate_sets, prediction_sets
from brokkr.shift.reliability import confidence_and_correct, ece, softmax
from brokkr.shift.selective import aurc, optimal_aurc

N_RESAMPLES, SEED = 1000, 0


def paired_ci(difference, n: int) -> list:
    """95% interval of difference(idx) over bootstrap resamples idx of the same n images."""
    rng = np.random.default_rng(SEED)
    values = [difference(rng.integers(0, n, n)) for _ in range(N_RESAMPLES)]
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def excludes_zero(ci: list) -> bool:
    return ci[0] > 0 or ci[1] < 0


def plain(value):
    """NumPy numbers and nested containers -> plain Python, so the verdicts can be saved as JSON."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


def result(parts: dict, numbers: dict, within_noise=None, notes=None) -> dict:
    parts = {k: bool(v) for k, v in parts.items()}
    return plain({"verdict": "PASS" if all(parts.values()) else "FAIL", "parts": parts, "numbers": numbers,
                  "within_noise": None if within_noise is None else bool(within_noise), "notes": notes or []})


def not_run(reason: str) -> dict:
    return {"verdict": "NOT RUN", "parts": {}, "numbers": {}, "within_noise": None, "notes": [reason]}


def top1_correct(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1.0 where the top answer is right; ties go to the lower class number, as everywhere."""
    return (np.argsort(-logits, axis=1, kind="stable")[:, 0] == labels).astype(np.float64)


def e_aurc(confidence: np.ndarray, correct: np.ndarray) -> float:
    return aurc(confidence, correct) - optimal_aurc(correct)


def h10(best: np.ndarray, labels: np.ndarray) -> dict:
    """Best INT8 clean test top-1 >= 67.9%."""
    top1 = float(top1_correct(best, labels).mean())
    return result({"top-1 >= 67.9%": top1 >= 0.679}, {"top1": top1})


def h11(best: np.ndarray, unrounded: np.ndarray, labels: np.ndarray, size_best: int, size_unrounded: int,
        e_aurc_noise: float) -> dict:
    """Unrounded output: 0 ties; E-AURC >= 10% lower with paired CI excluding zero; file < 5% bigger."""
    ties = int(((unrounded == unrounded.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())
    conf_b, corr_b = confidence_and_correct(best, labels)
    conf_u, corr_u = confidence_and_correct(unrounded, labels)
    e_b, e_u = e_aurc(conf_b, corr_b), e_aurc(conf_u, corr_u)
    diff = e_b - e_u
    ci = paired_ci(lambda i: e_aurc(conf_b[i], corr_b[i]) - e_aurc(conf_u[i], corr_u[i]), len(labels))
    relative = diff / e_b
    growth = size_unrounded / size_best - 1
    notes = []
    if relative < 0.15:
        notes.append("E-AURC improvement under 15%: within about 3x the build-to-build range "
                     f"({e_aurc_noise:.4f}), which comes from only 4 builds")
    return result({"0 tied test images": ties == 0,
                   "E-AURC at least 10% lower (relative)": relative >= 0.10,
                   "paired CI of the E-AURC difference excludes zero": excludes_zero(ci),
                   "file size increase under 5%": growth < 0.05},
                  {"tied_images": ties, "e_aurc_best": e_b, "e_aurc_unrounded": e_u,
                   "e_aurc_best_minus_unrounded": diff, "ci95": ci, "relative_improvement": relative,
                   "file_size_increase": growth},
                  within_noise=abs(diff) < e_aurc_noise, notes=notes)


def h12(best_dark5: np.ndarray, without_darkness_dark5: np.ndarray, labels: np.ndarray,
        top1_noise: float) -> dict:
    """At darkness s5, calibrated-without-darkness minus best INT8 >= 10 points, paired CI excl. zero."""
    a, b = top1_correct(best_dark5, labels), top1_correct(without_darkness_dark5, labels)
    diff = float(b.mean() - a.mean())
    ci = paired_ci(lambda i: b[i].mean() - a[i].mean(), len(labels))
    return result({"at least 10 points higher": diff >= 0.10,
                   "paired CI excludes zero": excludes_zero(ci)},
                  {"top1_best": float(a.mean()), "top1_without_darkness": float(b.mean()),
                   "difference": diff, "ci95": ci},
                  within_noise=abs(diff) < top1_noise)


def h13(best_clean: np.ndarray, leave_one_out_clean: list, labels: np.ndarray, top1_noise: float) -> dict:
    """Average clean top-1 of the five leave-one-out models within 1 point of best INT8 (judged on
    the measured value; the paired CI is reported, see the dated 3.7 note)."""
    a = top1_correct(best_clean, labels)
    b = np.mean([top1_correct(z, labels) for z in leave_one_out_clean], axis=0)  # per-image average
    diff = float(b.mean() - a.mean())
    ci = paired_ci(lambda i: b[i].mean() - a[i].mean(), len(labels))
    return result({"within 1 point (between -1 and +1)": -0.01 <= diff <= 0.01},
                  {"top1_best": float(a.mean()), "top1_leave_one_out_average": float(b.mean()),
                   "difference": diff, "ci95_reported_only": ci},
                  within_noise=abs(diff) < top1_noise)


def h14(temperature: float, clean: np.ndarray, severity5: dict, labels: np.ndarray) -> dict:
    """FP32: T < 1; clean ECE with scaling < 0.03; at severity 5, scaled ECE >= raw ECE + 0.02 with
    paired CI excluding zero, for at least 3 of 5 corruptions."""
    conf_scaled, correct = confidence_and_correct(clean / temperature, labels)
    clean_ece = ece(conf_scaled, correct)
    per_corruption, backfires = {}, 0
    for name, logits in severity5.items():
        raw, correct5 = confidence_and_correct(logits, labels)
        scaled, _ = confidence_and_correct(logits / temperature, labels)
        diff = ece(scaled, correct5) - ece(raw, correct5)
        ci = paired_ci(lambda i, s=scaled, r=raw, c=correct5: ece(s[i], c[i]) - ece(r[i], c[i]), len(labels))
        holds = diff >= 0.02 and excludes_zero(ci)
        backfires += holds
        per_corruption[name] = {"ece_raw": ece(raw, correct5), "ece_scaled": ece(scaled, correct5),
                                "scaled_minus_raw": diff, "ci95": ci, "holds": holds}
    return result({"T < 1": temperature < 1, "clean ECE with scaling < 0.03": clean_ece < 0.03,
                   "scaling backfires at severity 5 for at least 3 of 5": backfires >= 3},
                  {"temperature": temperature, "clean_ece_scaled": clean_ece,
                   "severity5": per_corruption, "corruptions_backfiring": backfires})


def set_stats(probs: np.ndarray, labels: np.ndarray, threshold: float) -> tuple:
    """Per image: 1.0 if the right class is in the set, and the set size."""
    sets = prediction_sets(probs, threshold)
    return sets[np.arange(len(labels)), labels].astype(np.float64), sets.sum(axis=1).astype(np.float64)


def h15(clean: np.ndarray, severity3: dict, labels: np.ndarray, clean_threshold: float,
        robust_thresholds: dict) -> dict:
    """FP32: coverage at severity 3 (held-out corruption, robust minus clean-tuned threshold, averaged
    over the five) >= 10 points with paired CI excluding zero; clean set size with the robust
    thresholds at least 2x, with paired CI of the difference excluding zero."""
    clean_probs = softmax(clean)
    cover_diff, per_corruption = [], {}
    for name, logits in severity3.items():
        probs = softmax(logits)
        cov_c, size_c = set_stats(probs, labels, clean_threshold)
        cov_r, size_r = set_stats(probs, labels, robust_thresholds[name])
        cover_diff.append(cov_r - cov_c)
        per_corruption[name] = {"coverage_clean_tuned": float(cov_c.mean()),
                                "set_size_clean_tuned": float(size_c.mean()),
                                "coverage_robust": float(cov_r.mean()),
                                "set_size_robust": float(size_r.mean())}
    d = np.mean(cover_diff, axis=0)  # per image, averaged over the five held-out corruptions
    cov_ci = paired_ci(lambda i: d[i].mean(), len(labels))
    _, size_clean = set_stats(clean_probs, labels, clean_threshold)
    size_robust = np.mean([set_stats(clean_probs, labels, t)[1] for t in robust_thresholds.values()], axis=0)
    size_ci = paired_ci(lambda i: size_robust[i].mean() - size_clean[i].mean(), len(labels))
    ratio = float(size_robust.mean() / size_clean.mean())
    return result({"coverage at least 10 points higher": d.mean() >= 0.10,
                   "paired CI of coverage difference excludes zero": excludes_zero(cov_ci),
                   "clean set size at least doubles": ratio >= 2,
                   "paired CI of set size difference excludes zero": excludes_zero(size_ci)},
                  {"coverage_gain_severity3": float(d.mean()), "coverage_gain_ci95": cov_ci,
                   "clean_set_size_clean_tuned": float(size_clean.mean()),
                   "clean_set_size_robust": float(size_robust.mean()), "clean_set_size_ratio": ratio,
                   "clean_set_size_difference_ci95": size_ci, "severity3": per_corruption})


def h16(threshold: float, clean: np.ndarray, harmful: dict, labels: np.ndarray) -> dict:
    """FP32 alarm: fires in >= 90% of the harmful windows and in at most 2 of the 100 clean windows."""
    def window_fires(logits):
        confidence, _ = confidence_and_correct(logits, labels)
        return fires(consecutive_window_means(confidence), threshold)

    clean_fires = int(window_fires(clean).sum())
    harmful_fires = np.concatenate([window_fires(z) for z in harmful.values()])
    rate = float(harmful_fires.mean())
    return result({"fires in >= 90% of harmful windows": rate >= 0.90,
                   "fires in at most 2 of 100 clean windows": clean_fires <= 2},
                  {"threshold": threshold, "clean_windows_firing": clean_fires,
                   "harmful_windows": len(harmful_fires), "harmful_fire_rate": rate},
                  notes=["The threshold was set on the tuning split, which is harder than test: it may sit "
                         "slightly low (fewer clean false alarms, a less sensitive alarm)."])


def h17(v2_logits, v2_labels, clean_threshold: float) -> dict:
    """FP32 on ImageNetV2: top-1 between 60% and 68%; upper end of clean-tuned coverage CI < 88%."""
    if v2_logits is None:
        return not_run("postponed to task 3.9 (ImageNetV2 not downloaded, licence not recorded)")
    top1 = float(top1_correct(v2_logits, v2_labels).mean())
    sets = evaluate_sets(softmax(v2_logits), v2_labels, clean_threshold, n_resamples=N_RESAMPLES, seed=SEED)
    return result({"top-1 between 60% and 68%": 0.60 <= top1 <= 0.68,
                   "upper end of coverage CI below 88%": sets["coverage_ci95"][1] < 0.88},
                  {"top1": top1, "coverage": sets["coverage"], "coverage_ci95": sets["coverage_ci95"],
                   "mean_set_size": sets["mean_set_size"]})
