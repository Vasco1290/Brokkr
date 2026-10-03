"""Choose the INT8 calibration method: highest clean top-1 on the tuning split (task 3.2).

Usage:  python scripts/11_choose_int8.py [--model NAME]
Needs:  results/accuracy/<model>_<candidate>_imagenet-1k-val_tuning.json for every candidate:
        int8 (the default, MinMax) and int8_<method> from scripts/10_int8_methods.py, each measured
        with scripts/03_evaluate_accuracy.py --precision <candidate> --split tuning
Writes: results/choices/<model>_int8_method.json

The rule was fixed before measuring (docs/hypotheses_stage3.md, design rules): the candidate with the
highest clean top-1 on the tuning split is "best INT8", and only it goes on to the later tasks.
Paired differences against MinMax (same images, 95% bootstrap interval) are printed and saved to show
how clear the choice is, but they do not change the rule. An exact tie is not covered by the rule,
so it stops with FAIL instead of picking one.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from brokkr_edge.accuracy import paired_bootstrap_diff
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.quantize import INT8_METHODS
from brokkr_edge.results import load_arrays, make_record, save_record

DATASET = "imagenet-1k-val"
# The default INT8 model ("int8") was built with MinMax; the other candidates are named after their method.
CANDIDATES = {"int8": "minmax", **{f"int8_{m}": m for m in INT8_METHODS if m != "minmax"}}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
args = parser.parse_args()

correct, positions = {}, {}
for precision in CANDIDATES:
    path = Path("results/accuracy") / f"{args.model}_{precision}_{DATASET}_tuning.json"
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run scripts/03_evaluate_accuracy.py --precision {precision} "
                 "--split tuning first.")
    arrays = load_arrays(path)
    top1 = np.argsort(-arrays["logits"], axis=1, kind="stable")[:, 0]  # same tie rule as everywhere
    correct[precision] = (top1 == arrays["labels"]).astype(float)
    positions[precision] = arrays["positions"]

same_images = all(np.array_equal(positions["int8"], p) for p in positions.values())
if not same_images:
    sys.exit("FAIL: the candidates were not measured on the same tuning images")

rows = []
print(f"Clean tuning split, {len(correct['int8']):,} images\n")
print(f"{'candidate':>22} {'method':>17} {'top-1':>8}   vs MinMax (paired 95% CI)")
for precision, method in CANDIDATES.items():
    diff, low, high = paired_bootstrap_diff(correct["int8"], correct[precision])
    rows.append({"precision": precision, "calibration_method": method,
                 "top1": float(correct[precision].mean()),
                 "diff_vs_minmax": diff, "diff_vs_minmax_ci95": [low, high]})
    change = f"{diff * 100:+.2f} points ({low * 100:+.2f} to {high * 100:+.2f})"
    change = "" if precision == "int8" else change
    print(f"{precision:>22} {method:>17} {correct[precision].mean():>8.2%}   {change}")

best_top1 = max(r["top1"] for r in rows)
winners = [r for r in rows if r["top1"] == best_top1]
if len(winners) > 1:
    print(f"\nFAIL: exact tie between {[w['precision'] for w in winners]}; the pre-registered rule does not "
          "cover ties. Decide (and record the decision) before going on.")
    sys.exit(1)
best = winners[0]
runner_up = max((r for r in rows if r is not best), key=lambda r: r["top1"])
diff, low, high = paired_bootstrap_diff(correct[runner_up["precision"]], correct[best["precision"]])
print(f"\nBest INT8: {best['precision']} ({best['calibration_method']}), top-1 {best['top1']:.2%}")
print(f"Margin over the runner-up ({runner_up['precision']}): {diff * 100:+.2f} points "
      f"(paired 95% CI {low * 100:+.2f} to {high * 100:+.2f})")

record = make_record("choice", args.model, best["precision"], {
    "settings": {"dataset": DATASET, "split": "tuning", "n_images": len(correct["int8"]),
                 "rule": "highest clean top-1 on the tuning split (docs/hypotheses_stage3.md)",
                 "bootstrap_resamples": 1000, "seed": 0},
    "metrics": {"chosen": best["precision"], "chosen_method": best["calibration_method"],
                "margin_over_runner_up": diff, "margin_over_runner_up_ci95": [low, high],
                "runner_up": runner_up["precision"]},
    "raw": {"candidates": rows},
}, machine_fingerprint())
out = save_record(record, Path("results/choices") / f"{args.model}_int8_method.json")
print(f"Saved to {out}\nPASS")
