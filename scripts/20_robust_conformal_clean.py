"""After the Stage 3 verdicts: what do the robust conformal thresholds do on CLEAN test images?

Usage:  python scripts/20_robust_conformal_clean.py [--model NAME]
Needs:  results/choices/<model>_shift_aware.json (task 3.6) and the clean test scores of FP32 and best
        INT8 (Stage 2 sweep and the final run)
Writes: results/final/<model>_robust_conformal_clean_after_verdicts.json

H15 judged the clean SET SIZE of the robust thresholds, not their clean coverage. This reports both,
for each held-out-corruption threshold and on average, so coverage is never shown without set size.
Computed after the verdicts from saved scores (no new measurement); not a prediction.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from brokkr.fingerprint import machine_fingerprint
from brokkr.results import load_arrays, make_record, save_record
from brokkr.shift import CORRUPTIONS
from brokkr.shift.conformal import prediction_sets
from brokkr.shift.reliability import softmax

DATASET = "imagenet-1k-val"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
args = parser.parse_args()
m = args.model

best = json.loads((Path("results/choices") / f"{m}_int8_method.json").read_text())["metrics"]["chosen"]
shift = json.loads((Path("results/choices") / f"{m}_shift_aware.json").read_text())
robust = shift["metrics"]["robust_conformal"]

results = {}
for precision in ("fp32", best):
    arrays = load_arrays(Path("results/sweep") / f"{m}_{precision}_{DATASET}_test_clean_s0.json")
    probs, labels = softmax(arrays["logits"]), arrays["labels"]
    rows = {}
    thresholds = {"clean-tuned": robust[precision]["clean_only_threshold"]}
    for c in CORRUPTIONS:
        thresholds[f"robust, without {c}"] = robust[precision]["held_out"][c]["threshold"]
    for name, threshold in thresholds.items():
        sets = prediction_sets(probs, threshold)
        rows[name] = {"coverage": float(sets[np.arange(len(labels)), labels].mean()),
                      "mean_set_size": float(sets.sum(axis=1).mean())}
    robust_rows = [r for name, r in rows.items() if name.startswith("robust")]
    rows["robust, average of the five"] = {k: float(np.mean([r[k] for r in robust_rows]))
                                           for k in ("coverage", "mean_set_size")}
    results[precision] = rows
    print(f"\n{precision}, clean test images ({len(labels):,}): coverage (average set size)")
    for name, r in rows.items():
        print(f"    {name:<30} {r['coverage']:.2%} ({r['mean_set_size']:.2f})")

record = make_record("after_verdicts", m, "all", {
    "settings": {"label": "computed after the Stage 3 verdicts from saved scores; not a prediction",
                 "split": "test", "condition": "clean", "scores": "raw"},
    "metrics": results,
}, machine_fingerprint())
out = save_record(record, Path("results/final") / f"{m}_robust_conformal_clean_after_verdicts.json")
print(f"\nSaved to {out}")
