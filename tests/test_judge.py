"""The Stage 3 judging rules on fake data, where the right verdict is known in advance."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from brokkr import judge
from brokkr.results import save_arrays

N, CLASSES = 2_000, 10
rng = np.random.default_rng(0)
LABELS = rng.integers(0, CLASSES, N)


def scores_with_accuracy(accuracy: float, seed: int = 1, noise: float = 1.0) -> np.ndarray:
    """Fake logits whose top answer is right for the first `accuracy` share of images."""
    r = np.random.default_rng(seed)
    logits = r.normal(0, noise, (N, CLASSES))
    right = np.arange(N) < int(accuracy * N)
    wrong_class = (LABELS + 1) % CLASSES
    logits[np.arange(N), np.where(right, LABELS, wrong_class)] += 5
    return logits


def calibrated(seed: int = 2, scale: float = 2.0, n: int = 20_000):
    """Fake logits and labels where softmax(logits) is honest (labels drawn from it).

    Many images, because ECE is biased upwards on small samples."""
    r = np.random.default_rng(seed)
    logits = r.normal(0, scale, (n, CLASSES))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = np.array([r.choice(CLASSES, p=p) for p in probs])
    return logits, labels


def test_paired_ci_of_a_constant_difference_is_that_constant():
    assert judge.paired_ci(lambda i: 0.5, N) == [0.5, 0.5]
    assert judge.ci_side([0.1, 0.3]) == "above zero"
    assert judge.ci_side([-0.3, -0.1]) == "below zero"
    assert judge.ci_side([-0.1, 0.1]) == "includes zero"


def test_an_interval_on_the_wrong_side_does_not_count():
    """The final run's H11 case: unrounded slightly WORSE, interval entirely below zero."""
    best = scores_with_accuracy(0.7)
    right = np.arange(N) < int(0.7 * N)
    best[right, LABELS[right]] += 3  # best ranks its mistakes well
    worse = scores_with_accuracy(0.7)  # same accuracy, blurrier ranking
    r = judge.h11(best, worse, LABELS, size_best=100, size_unrounded=100, e_aurc_noise=0.0026)
    assert r["numbers"]["ci_side"] == "below zero"
    assert not r["parts"]["paired CI of the E-AURC difference above zero (unrounded better)"]
    assert r["verdict"] == "FAIL"


def test_h10():
    assert judge.h10(scores_with_accuracy(0.70), LABELS)["verdict"] == "PASS"
    assert judge.h10(scores_with_accuracy(0.60), LABELS)["verdict"] == "FAIL"


def test_h11_passes_when_unrounding_removes_ties_and_ranks_mistakes_better():
    unrounded = scores_with_accuracy(0.7)
    right = np.arange(N) < int(0.7 * N)
    unrounded[right, LABELS[right]] += 3  # right answers are clearly more confident: good ranking
    best = np.round(unrounded)  # rounding: many ties, blurrier ranking
    best[np.arange(N), (LABELS + 3) % CLASSES] = best.max(axis=1)  # force a tie on every image
    r = judge.h11(best, unrounded, LABELS, size_best=100, size_unrounded=101, e_aurc_noise=0.0026)
    assert r["numbers"]["tied_images"] == 0
    assert r["verdict"] == "PASS" and r["within_noise"] is False


def test_h11_fails_and_is_within_noise_when_nothing_changes():
    same = scores_with_accuracy(0.7)
    r = judge.h11(same, same, LABELS, size_best=100, size_unrounded=100, e_aurc_noise=0.0026)
    assert r["verdict"] == "FAIL" and r["within_noise"] is True
    assert any("15%" in note for note in r["notes"])
    too_big = judge.h11(same, same, LABELS, size_best=100, size_unrounded=106, e_aurc_noise=0.0026)
    assert not too_big["parts"]["file size increase under 5%"]


def test_h12():
    passes = judge.h12(scores_with_accuracy(0.2), scores_with_accuracy(0.4), LABELS, top1_noise=0.0028)
    assert passes["verdict"] == "PASS" and passes["numbers"]["difference"] == pytest.approx(0.2)
    fails = judge.h12(scores_with_accuracy(0.2), scores_with_accuracy(0.25), LABELS, top1_noise=0.0028)
    assert fails["verdict"] == "FAIL"


def test_h13_is_judged_on_the_measured_difference():
    best = scores_with_accuracy(0.70)
    close = [scores_with_accuracy(a, seed=s) for s, a in enumerate([0.69, 0.70, 0.71, 0.70, 0.70])]
    r = judge.h13(best, close, LABELS, top1_noise=0.0028)
    assert r["verdict"] == "PASS" and r["numbers"]["difference"] == pytest.approx(0.0, abs=0.001)
    far = [scores_with_accuracy(0.67, seed=s) for s in range(5)]
    assert judge.h13(best, far, LABELS, top1_noise=0.0028)["verdict"] == "FAIL"


def test_h14_honest_model_sharpened_backfires():
    clean, labels = calibrated(seed=2)
    damaged = {f"c{k}": calibrated(seed=10 + k)[0] for k in range(5)}
    # These fake images reuse `labels`, so the "damaged" scores are wrong-ish and sharpening hurts.
    r = judge.h14(0.5, clean, damaged, labels)
    assert r["parts"]["T < 1"]
    assert not r["parts"]["clean ECE with scaling < 0.03"]  # an honest model made over-confident
    honest = judge.h14(1.0, clean, damaged, labels)
    assert honest["parts"]["clean ECE with scaling < 0.03"] and not honest["parts"]["T < 1"]
    assert honest["verdict"] == "FAIL"


def test_h15_robust_thresholds_raise_coverage_at_the_cost_of_set_size():
    clean = scores_with_accuracy(0.9)
    severe = {f"c{k}": scores_with_accuracy(0.3, seed=k) for k in range(5)}
    robust = {name: 0.999 for name in severe}  # almost every class goes in the set
    r = judge.h15(clean, severe, LABELS, clean_threshold=0.5, robust_thresholds=robust)
    assert r["verdict"] == "PASS"
    for numbers in r["numbers"]["severity3"].values():  # coverage always comes with set size
        pairs = {"coverage_robust", "set_size_robust", "coverage_clean_tuned", "set_size_clean_tuned"}
        assert pairs <= set(numbers)
    same = judge.h15(clean, severe, LABELS, clean_threshold=0.5, robust_thresholds={n: 0.5 for n in severe})
    assert same["verdict"] == "FAIL"


def test_h16():
    labels = np.zeros(10_000, dtype=int)
    confident = np.zeros((10_000, CLASSES))
    confident[:, 0] = 5.0
    unsure = np.zeros((10_000, CLASSES))
    threshold = 0.5
    r = judge.h16(threshold, confident, {"a": unsure, "b": unsure}, labels)
    assert r["verdict"] == "PASS" and r["numbers"]["harmful_windows"] == 200
    assert judge.h16(threshold, confident, {"a": confident}, labels)["verdict"] == "FAIL"


def test_h17_is_not_run_without_imagenetv2():
    assert judge.h17(None, None, 0.96)["verdict"] == "NOT RUN"
    r = judge.h17(scores_with_accuracy(0.65), LABELS, clean_threshold=0.5)
    assert r["parts"]["top-1 between 60% and 68%"]


def test_verdicts_can_be_saved_as_json():
    clean = scores_with_accuracy(0.9)
    severe = {f"c{k}": scores_with_accuracy(0.3, seed=k) for k in range(5)}
    r = judge.h15(clean, severe, LABELS, clean_threshold=0.5, robust_thresholds={n: 0.999 for n in severe})
    assert json.loads(json.dumps(r))["verdict"] == "PASS"


def test_judging_script_end_to_end_on_a_fake_results_folder(tmp_path):
    """Run scripts/18_judge_stage3.py on small fake files laid out like the real final run."""
    from brokkr.shift import CORRUPTIONS

    m, best, n = "fake", "int8_x", 1_000
    labels = np.random.default_rng(0).integers(0, CLASSES, n)

    def write(path, content):
        path = tmp_path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))

    def scores(precision, condition, accuracy):
        r = np.random.default_rng(len(precision) + len(condition))
        logits = r.normal(0, 1, (n, CLASSES)).astype(np.float32)
        right = r.random(n) < accuracy
        logits[np.arange(n), np.where(right, labels, (labels + 1) % CLASSES)] += 4
        record = {"kind": "accuracy"}
        path = tmp_path / "results/sweep" / f"{m}_{precision}_imagenet-1k-val_test_{condition}.json"
        save_arrays(record, path, logits=logits, labels=labels, positions=np.arange(n))
        path.write_text(json.dumps(record))

    write(f"results/choices/{m}_int8_method.json", {"metrics": {"chosen": best}})
    write(f"results/checks/{m}_{best}_calibration_luck.json",
          {"metrics": {"top1_range": 0.0028, "e_aurc_range": 0.0026}})
    write(f"results/choices/{m}_temperatures.json", {"metrics": {"temperatures": {"fp32": 0.74}}})
    robust = {"clean_only_threshold": 0.6, "held_out": {c: {"threshold": 0.99} for c in CORRUPTIONS}}
    write(f"results/choices/{m}_shift_aware.json",
          {"metrics": {"robust_conformal": {"fp32": robust, best: robust},
                       "alarm": {"fp32": {"threshold": 0.5}}}})
    for precision in (best, f"{best}_unrounded"):
        write(f"models/{m}_{precision}.json", {"file": {"size_bytes": 1000}})
    for c in CORRUPTIONS:
        for s in (1, 2, 3, 4, 5):
            scores("fp32", f"{c}_s{s}", 0.3)
        scores(best, f"{c}_s3", 0.2)
        scores(f"{best}_mixed_without_{c}", "clean_s0", 0.7)
    for precision in ("fp32", best, f"{best}_unrounded"):
        scores(precision, "clean_s0", 0.75)
    scores(best, "darkness_s5", 0.2)
    scores(f"{best}_mixed_without_darkness", "darkness_s5", 0.4)

    script = Path(__file__).parents[1] / "scripts" / "18_judge_stage3.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}
    run = subprocess.run([sys.executable, str(script), "--model", m], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    saved = json.loads((tmp_path / f"results/final/{m}_stage3_verdicts.json").read_text())["metrics"]
    assert saved["H17"]["verdict"] == "NOT RUN"
    judged = ("H10", "H11", "H12", "H13", "H14", "H15", "H16")
    assert all(saved[h]["verdict"] in ("PASS", "FAIL") for h in judged)
    assert saved["H12"]["verdict"] == "PASS"  # 40% vs 20% at darkness s5

