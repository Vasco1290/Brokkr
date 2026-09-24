"""Compute reliability numbers from saved accuracy results (no model is re-run).

Usage:  python scripts/07_reliability.py
Needs:  results/accuracy/*.json with their .npz logits (scripts/03_evaluate_accuracy.py)
Writes: results/reliability/<same name>_calibration.json for each accuracy result

Calibration: expected calibration error (ECE) with a bootstrap 95% CI, average confidence vs
accuracy, and per-bin data for a reliability diagram.
"""

import json
import sys
from pathlib import Path

from brokkr.fingerprint import machine_fingerprint
from brokkr.results import load_arrays, make_record, save_record
from brokkr.shift.reliability import calibration

sources = sorted(p for p in Path("results/accuracy").glob("*.json")
                 if "raw_arrays" in json.loads(p.read_text()))
if not sources:
    sys.exit("FAIL: no accuracy results with saved logits. Run scripts/03_evaluate_accuracy.py first.")

machine = machine_fingerprint()
passed = True
print(f"{'result':48s} {'ECE':>7} {'95% CI':>17} {'confidence':>11} {'accuracy':>9}")
for source in sources:
    acc = json.loads(source.read_text())
    arrays = load_arrays(source)  # refuses a logits file whose checksum doesn't match
    cal = calibration(arrays["logits"], arrays["labels"])

    result = {
        "settings": {**{k: acc["settings"][k] for k in ("dataset", "n_images", "split")
                        if k in acc["settings"]},
                     "source_result": source.name, "source_arrays_sha256": acc["raw_arrays"]["sha256"],
                     "bootstrap_resamples": 1000},
        "metrics": {k: v for k, v in cal.items() if k != "bins"},
        "raw": {"bins": cal["bins"]},
    }
    record = make_record("calibration", acc["model"], acc["precision"], result, machine)
    save_record(record, Path("results/reliability") / f"{source.stem}_calibration.json")

    lo, hi = cal["ece_ci95"]
    print(f"{source.stem:48s} {cal['ece']:7.4f}   ({lo:.4f}-{hi:.4f}) {cal['mean_confidence']:11.2%} "
          f"{cal['accuracy']:9.2%}")
    passed &= 0 <= cal["ece"] <= 1 and lo <= cal["ece"] <= hi
    passed &= sum(b["count"] for b in cal["bins"]) == len(arrays["labels"])
    passed &= abs(cal["accuracy"] - acc["metrics"]["top1"]) < 1e-9  # same answers as the accuracy run

print("\nPASS" if passed else "\nFAIL")
sys.exit(0 if passed else 1)
