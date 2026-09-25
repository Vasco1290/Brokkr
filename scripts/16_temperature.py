"""Fit one temperature T per model on the clean tuning split (task 3.5).

Usage:  python scripts/16_temperature.py [--model NAME]
Needs:  results/accuracy/<model>_<precision>_imagenet-1k-val_tuning.json (+ .npz) for every model
        in the final run (scripts/03_evaluate_accuracy.py --precision <p> --split tuning), and
        results/choices/<model>_int8_method.json (scripts/11)
Writes: results/choices/<model>_temperatures.json

Rules fixed before fitting (docs/hypotheses_stage3.md, dated notes): T minimises the negative
log-likelihood of the true class on the clean tuning images; golden-section search over T = 0.1 to
10 (brokkr.shift.reliability.fit_temperature). If any T lands within 1% of either end of that
range, the run stops and reports it instead of saving anything.

Checks: NLL at T is not above NLL at T = 1, and no image's top answer changes.
Tool check (not a result): ECE on the same tuning images before and after scaling, with Stage 2's
15-bin ECE. H14 is judged on the TEST split in the final run, task 3.7.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from brokkr.fingerprint import machine_fingerprint
from brokkr.results import load_arrays, make_record, save_record
from brokkr.shift import CORRUPTIONS
from brokkr.shift.reliability import confidence_and_correct, ece, fit_temperature, nll

DATASET = "imagenet-1k-val"
LOW, HIGH = 0.1, 10.0
EDGE = 0.01  # "at the edge" = within 1% of either end of the range

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
args = parser.parse_args()

choice = json.loads((Path("results/choices") / f"{args.model}_int8_method.json").read_text())
best = choice["metrics"]["chosen"]
precisions = (["fp32", "fp16", "int8", best, f"{best}_unrounded"]
              + [f"{best}_mixed_without_{c}" for c in CORRUPTIONS])

paths = {p: Path("results/accuracy") / f"{args.model}_{p}_{DATASET}_tuning.json" for p in precisions}
missing = [p for p, path in paths.items() if not path.exists()]
if missing:
    sys.exit(f"FAIL: no clean tuning result for {missing}. Run scripts/03_evaluate_accuracy.py "
             "--precision <p> --split tuning first.")

rows = {}
print(f"Clean tuning split; T searched in [{LOW}, {HIGH}]\n")
print(f"{'model':<44}{'T':>7}{'NLL before':>12}{'after':>8}{'ECE before':>12}{'after':>8}")
for precision, path in paths.items():
    arrays = load_arrays(path)
    logits, labels = arrays["logits"], arrays["labels"]
    t = fit_temperature(logits, labels, low=LOW, high=HIGH)
    if t < LOW * (1 + EDGE) or t > HIGH * (1 - EDGE):
        print(f"\nFAIL: {precision} fitted T = {t:.4f}, at the edge of [{LOW}, {HIGH}]. "
              "Stopped; nothing saved.")
        sys.exit(1)
    scaled = logits.astype(np.float64) / t
    before, after = confidence_and_correct(logits, labels), confidence_and_correct(scaled, labels)
    row = {"temperature": t, "nll_before": nll(logits, labels), "nll_after": nll(logits, labels, t),
           "ece_before": ece(*before), "ece_after": ece(*after),
           "mean_confidence_before": float(before[0].mean()), "mean_confidence_after": float(after[0].mean()),
           "top1": float(before[1].mean()), "n_images": len(labels)}
    same_answers = np.array_equal(logits.argmax(1), scaled.argmax(1))
    if row["nll_after"] > row["nll_before"] or not same_answers:
        print(f"\nFAIL: {precision}: NLL rose ({row['nll_before']:.4f} -> {row['nll_after']:.4f}) "
              f"or top answers changed ({not same_answers}). Stopped; nothing saved.")
        sys.exit(1)
    rows[precision] = row
    print(f"{precision:<44}{t:>7.4f}{row['nll_before']:>12.4f}{row['nll_after']:>8.4f}"
          f"{row['ece_before']:>12.4f}{row['ece_after']:>8.4f}")

record = make_record("choice", args.model, "all", {
    "settings": {"dataset": DATASET, "split": "tuning", "fit": "minimise NLL of the true class",
                 "search": "golden-section over log T", "range": [LOW, HIGH], "tolerance_log_t": 1e-4,
                 "edge_check": f"stop if within {EDGE:.0%} of either end", "ece_bins": 15},
    "metrics": {"temperatures": {p: r["temperature"] for p, r in rows.items()}},
    "raw": {"per_model": rows},
}, machine_fingerprint())
out = save_record(record, Path("results/choices") / f"{args.model}_temperatures.json")
print(f"\nSaved to {out}\nPASS")
