"""EXPLORATORY, computed after the Stage 3 verdicts: alarm firing rate for every test condition.

Usage:  python scripts/19_alarm_all_conditions.py [--model NAME]
Needs:  the final run's test results (scripts/run_stage3_final.py), the Stage 2 test sweep, and
        results/choices/<model>_shift_aware.json (alarm thresholds, task 3.6)
Writes: results/final/<model>_alarm_all_conditions_exploratory.json

H16 was judged on FP32 over the 12 pre-declared harmful conditions and clean images only. This table
shows the same alarm (same thresholds, same 100 windows of 100 test images per condition, seed 3) for
every condition each model was measured on: all 26 for FP32, FP16, default INT8, best INT8 and best
INT8 unrounded; clean plus the held-out corruption for the five leave-one-out models (the design
rule; nothing new was run). No new test measurement: it only re-reads saved scores. It is not a
prediction and changes no verdict.
"""

import argparse
import json
from pathlib import Path

from brokkr.fingerprint import machine_fingerprint
from brokkr.results import load_arrays, make_record, save_record
from brokkr.shift import CORRUPTIONS
from brokkr.shift.alarm import consecutive_window_means, fires
from brokkr.shift.reliability import confidence_and_correct

DATASET = "imagenet-1k-val"
HARMFUL = ({("defocus_blur", s) for s in (2, 3, 4, 5)} | {("motion_blur", s) for s in (2, 3, 4, 5)}
           | {("noise", s) for s in (3, 4, 5)} | {("fog", 5)})

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
args = parser.parse_args()
m = args.model

best = json.loads((Path("results/choices") / f"{m}_int8_method.json").read_text())["metrics"]["chosen"]
alarms = json.loads((Path("results/choices") / f"{m}_shift_aware.json").read_text())["metrics"]["alarm"]
thresholds = {p: a["threshold"] for p, a in alarms.items()}
all_conditions = [("clean", 0)] + [(c, s) for c in CORRUPTIONS for s in (1, 2, 3, 4, 5)]
models = {p: all_conditions for p in ("fp32", "fp16", "int8", best, f"{best}_unrounded")}
models.update({f"{best}_mixed_without_{c}": [("clean", 0)] + [(c, s) for s in (1, 2, 3, 4, 5)]
               for c in CORRUPTIONS})

rates = {}
for precision, conditions in models.items():
    rates[precision] = {}
    for corruption, severity in conditions:
        path = Path("results/sweep") / f"{m}_{precision}_{DATASET}_test_{corruption}_s{severity}.json"
        arrays = load_arrays(path)
        confidence, _ = confidence_and_correct(arrays["logits"], arrays["labels"])
        windows = fires(consecutive_window_means(confidence), thresholds[precision])
        rates[precision][f"{corruption} s{severity}"] = float(windows.mean())
    print(f"done: {precision}", flush=True)

short = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8 default", best: "best INT8",
         f"{best}_unrounded": "unrounded"}
full = [p for p in models if models[p] is all_conditions]
print("\nEXPLORATORY (after the verdicts): % of the 100 test windows where the alarm fires")
print(f"{'condition':<18}" + "".join(f"{short[p]:>13}" for p in full) + "   harmful (H16)?")
for corruption, severity in all_conditions:
    key = f"{corruption} s{severity}"
    marker = "yes" if (corruption, severity) in HARMFUL else ""
    print(f"{key:<18}" + "".join(f"{rates[p][key]:>12.0%} " for p in full) + f"   {marker}")
print("\nLeave-one-out INT8 models (clean and their held-out corruption only):")
for c in CORRUPTIONS:
    p = f"{best}_mixed_without_{c}"
    print(f"  without {c:<13}" + "  ".join(f"{k}: {v:.0%}" for k, v in rates[p].items()))

record = make_record("exploratory", m, "all", {
    "settings": {"label": "EXPLORATORY, computed after the Stage 3 verdicts; not a prediction",
                 "split": "test", "windows": "100 consecutive non-overlapping windows of 100 images, seed 3",
                 "thresholds": "results/choices/*_shift_aware.json (alarm, task 3.6)",
                 "harmful_conditions_h16": sorted(f"{c} s{s}" for c, s in HARMFUL)},
    "metrics": {"fire_rate": rates},
}, machine_fingerprint())
out = save_record(record, Path("results/final") / f"{m}_alarm_all_conditions_exploratory.json")
print(f"\nSaved to {out}")
