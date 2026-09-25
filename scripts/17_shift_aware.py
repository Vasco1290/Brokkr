"""Compute the shift-aware "I'm not sure" settings (task 3.6): robust conformal and alarm thresholds.

Usage:  python scripts/17_shift_aware.py [--model NAME]
Needs:  - conformal_calibration results: clean (scripts/03 --split conformal_calibration) and damaged
          (scripts/08 --split conformal_calibration) for FP32, FP16 and default INT8
        - clean tuning results (scripts/03 --split tuning) for every final-run model
        - results/choices/<model>_int8_method.json (scripts/11)
Writes: results/choices/<model>_shift_aware.json

Rules fixed before computing (docs/hypotheses_stage3.md, design rules and dated notes). All scores
are RAW (no temperature).

Robust conformal (FP32, FP16, default INT8): for each held-out corruption, the 5,000
conformal_calibration images are one-third clean and two-thirds damaged with the four other
corruptions at severities 1-5, balanced (brokkr.shift.robust_conformal, seed 9). The threshold is
the same LAC threshold as Stage 2, only on these images. Check: the clean-only threshold recomputed
here must equal Stage 2's saved one exactly.

Alarm (every final-run model): threshold = 1st percentile of the average confidence of 10,000
random windows of 100 clean tuning images (brokkr.shift.alarm, seed 4).

Tool checks (not results): coverage on the calibration mix itself is at least 90%, and about 1% of
the clean tuning windows fire. Coverage and alarm rates on the TEST split are measured in task 3.7.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from brokkr.fingerprint import machine_fingerprint
from brokkr.results import load_arrays, make_record, save_record
from brokkr.shift import CORRUPTIONS
from brokkr.shift.alarm import alarm_threshold, fires, random_window_means
from brokkr.shift.conformal import conformal_threshold, evaluate_sets
from brokkr.shift.reliability import confidence_and_correct, softmax
from brokkr.shift.robust_conformal import robust_calibration_plan

DATASET = "imagenet-1k-val"
COVERAGE = 0.9
PLAN_SEED, ALARM_SEED = 9, 4
CONFORMAL_MODELS = ["fp32", "fp16", "int8"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
args = parser.parse_args()

choice = json.loads((Path("results/choices") / f"{args.model}_int8_method.json").read_text())
best = choice["metrics"]["chosen"]
alarm_models = (["fp32", "fp16", "int8", best, f"{best}_unrounded"]
                + [f"{best}_mixed_without_{c}" for c in CORRUPTIONS])
passed = True

# 1. Robust conformal thresholds.
print("Robust conformal: calibration images one-third clean, two-thirds damaged (held-out type excluded)")
robust = {}
for precision in CONFORMAL_MODELS:
    clean_path = Path("results/accuracy") / f"{args.model}_{precision}_{DATASET}_conformal_calibration.json"
    clean = load_arrays(clean_path)
    labels, positions = clean["labels"], clean["positions"]

    # Check: the clean-only threshold must equal the one Stage 2 used.
    stage2 = json.loads((Path("results/reliability") /
                         f"{args.model}_{precision}_{DATASET}_test_conformal.json").read_text())
    clean_threshold = conformal_threshold(softmax(clean["logits"]), labels, COVERAGE)
    same_as_stage2 = clean_threshold == stage2["settings"]["threshold"]
    passed &= same_as_stage2
    print(f"\n{precision}: clean-only threshold {clean_threshold:.6f}, equal to Stage 2's: {same_as_stage2}")

    # Scores for every damaged version of the conformal_calibration images (task 3.1 sweep).
    damaged = {}
    for name in CORRUPTIONS:
        for severity in (1, 2, 3, 4, 5):
            arrays = load_arrays(Path("results/sweep") / f"{args.model}_{precision}_{DATASET}_"
                                 f"conformal_calibration_{name}_s{severity}.json")
            same_images = (np.array_equal(arrays["positions"], positions)
                           and np.array_equal(arrays["labels"], labels))
            if not same_images:
                sys.exit(f"FAIL: {precision} {name} s{severity} is not on the same images")
            damaged[(name, severity)] = arrays["logits"]

    robust[precision] = {"clean_only_threshold": clean_threshold, "held_out": {}}
    for held_out in CORRUPTIONS:
        allowed = [c for c in CORRUPTIONS if c != held_out]
        plan = robust_calibration_plan(len(labels), allowed, seed=PLAN_SEED)
        mixed = clean["logits"].copy()
        for pair in Counter(plan):
            if pair[0] is not None:
                rows = [i for i, p in enumerate(plan) if p == pair]
                mixed[rows] = damaged[pair][rows]
        probs = softmax(mixed)
        threshold = conformal_threshold(probs, labels, COVERAGE)
        on_mix = evaluate_sets(probs, labels, threshold, n_resamples=1)
        passed &= on_mix["coverage"] >= COVERAGE
        robust[precision]["held_out"][held_out] = {
            "threshold": threshold, "coverage_on_calibration_mix": on_mix["coverage"],
            "mean_set_size_on_calibration_mix": on_mix["mean_set_size"],
            "n_clean": sum(p[0] is None for p in plan), "n_damaged": sum(p[0] is not None for p in plan)}
        print(f"    without {held_out:<13} threshold {threshold:.6f}  (on its own calibration mix: "
              f"coverage {on_mix['coverage']:.2%}, set size {on_mix['mean_set_size']:.2f})")

# 2. Alarm thresholds.
print("\nAlarm: 1st percentile of 10,000 random 100-image windows of clean tuning images (raw confidence)")
alarms = {}
for precision in alarm_models:
    arrays = load_arrays(Path("results/accuracy") / f"{args.model}_{precision}_{DATASET}_tuning.json")
    confidence, _ = confidence_and_correct(arrays["logits"], arrays["labels"])
    threshold = alarm_threshold(confidence, seed=ALARM_SEED)
    share = float(fires(random_window_means(confidence, seed=ALARM_SEED), threshold).mean())
    passed &= 0.005 <= share <= 0.015
    alarms[precision] = {"threshold": threshold, "mean_confidence_clean_tuning": float(confidence.mean()),
                         "share_of_tuning_windows_firing": share}
    print(f"    {precision:<48} threshold {threshold:.4f}  (mean confidence {confidence.mean():.4f}; "
          f"{share:.2%} of tuning windows fire)")

record = make_record("choice", args.model, "all", {
    "settings": {"dataset": DATASET, "scores": "raw (no temperature)", "target_coverage": COVERAGE,
                 "conformal_method": "LAC, k-th smallest score, k = ceil((n+1) x 0.9)",
                 "robust_calibration": {"split": "conformal_calibration", "clean_fraction": "1/3",
                                        "plan_seed": PLAN_SEED, "severities": [1, 2, 3, 4, 5]},
                 "alarm": {"split": "tuning", "window": 100, "n_windows": 10_000, "percentile": 1,
                           "seed": ALARM_SEED, "fires": "window average strictly below threshold"}},
    "metrics": {"robust_conformal": robust, "alarm": alarms},
}, machine_fingerprint())
out = save_record(record, Path("results/choices") / f"{args.model}_shift_aware.json")
print(f"\nSaved to {out}")
print("PASS" if passed else "FAIL: a threshold differs from Stage 2, or a tool check failed")
sys.exit(0 if passed else 1)
