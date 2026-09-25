"""Compute reliability numbers from saved accuracy results (no model is re-run).

Usage:  python scripts/07_reliability.py [--sources results/accuracy results/sweep] [--out DIR]
Needs:  test-split results with .npz logits: results/accuracy/*_test.json (scripts/03) and
        results/sweep/*.json (scripts/08, corrupted images); for conformal sets also
        results/accuracy/<model>_<precision>_<dataset>_conformal_calibration.json
Writes: <out>/<source name>_calibration.json, _selective.json, _conformal.json
        (default out: results/reliability)

Calibration: expected calibration error (ECE) with a bootstrap 95% CI, average confidence vs
accuracy, and per-bin data for a reliability diagram.
Conformal: a threshold tuned on CLEAN conformal_calibration images for 90% coverage, then coverage
and set size on the test split (clean or corrupted: corrupted images test whether the promise
survives when test images no longer look like the calibration images).
Selective prediction: the risk-coverage curve, AURC and E-AURC with bootstrap 95% CIs, and the
error rate when answering only the most confident 80% / 50% of images.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from brokkr.fingerprint import machine_fingerprint
from brokkr.results import load_arrays, make_record, save_record
from brokkr.shift.conformal import conformal_threshold, evaluate_sets
from brokkr.shift.reliability import calibration, confidence_and_correct, softmax
from brokkr.shift.selective import selective_prediction

TARGET_COVERAGE = 0.90


def load(path: Path) -> tuple:
    """The JSON record and its arrays (the checksum is verified on loading)."""
    return json.loads(path.read_text()), load_arrays(path)


parser = argparse.ArgumentParser()
parser.add_argument("--sources", nargs="+", default=["results/accuracy", "results/sweep"])
parser.add_argument("--out", default="results/reliability")
args = parser.parse_args()
OUT = Path(args.out)


def is_test_result_with_logits(path: Path) -> bool:
    record = json.loads(path.read_text())
    return record.get("settings", {}).get("split") == "test" and "raw_arrays" in record


tests = sorted(p for folder in args.sources for p in Path(folder).glob("*.json")
               if is_test_result_with_logits(p))
if not tests:
    sys.exit("FAIL: no test-split results with saved logits. Run scripts/03_evaluate_accuracy.py first.")

machine = machine_fingerprint()
passed = True
for test_path in tests:
    acc, arrays = load(test_path)
    stem = test_path.stem
    corruption = {k: acc["settings"][k] for k in ("corruption", "severity") if k in acc["settings"]}
    base_settings = {"dataset": acc["settings"]["dataset"], "split": "test", **corruption,
                     "n_images": acc["settings"]["n_images"], "source_result": test_path.name,
                     "source_arrays_sha256": acc["raw_arrays"]["sha256"], "bootstrap_resamples": 1000}
    label = f"{corruption['corruption']} s{corruption['severity']}" if corruption else "clean"
    print(f"\n{acc['model']} {acc['precision']} {label} ({acc['settings']['n_images']:,} test images)")

    # Calibration
    cal = calibration(arrays["logits"], arrays["labels"])
    record = make_record("calibration", acc["model"], acc["precision"],
                         {"settings": base_settings, "metrics": {k: v for k, v in cal.items() if k != "bins"},
                          "raw": {"bins": cal["bins"]}}, machine)
    save_record(record, OUT / f"{stem}_calibration.json")
    lo, hi = cal["ece_ci95"]
    print(f"  ECE {cal['ece']:.4f} ({lo:.4f}-{hi:.4f}); mean confidence {cal['mean_confidence']:.2%} "
          f"vs accuracy {cal['accuracy']:.2%}")
    # Not "lo <= ece <= hi": ECE is biased upwards, so for near-zero ECE the bootstrap interval can
    # sit entirely above it (see brokkr/shift/reliability.py).
    passed &= 0 <= cal["ece"] <= 1 and 0 <= lo <= hi <= 1
    passed &= abs(cal["accuracy"] - acc["metrics"]["top1"]) < 1e-9

    # Selective prediction
    confidence, correct = confidence_and_correct(arrays["logits"], arrays["labels"])
    sel = selective_prediction(confidence, correct)
    curve = sel.pop("curve")
    record = make_record("selective", acc["model"], acc["precision"],
                         {"settings": {**base_settings, "confidence": "probability of the top answer"},
                          "metrics": sel, "raw": {"risk_coverage_curve": curve}}, machine)
    save_record(record, OUT / f"{stem}_selective.json")
    alo, ahi = sel["aurc_ci95"]
    print(f"  selective: AURC {sel['aurc']:.4f} ({alo:.4f}-{ahi:.4f}), E-AURC {sel['e_aurc']:.4f}; "
          f"error at 100% / 80% / 50% answered: {sel['risk_at_full_coverage']:.1%} / "
          f"{sel['risk_at_80pct_coverage']:.1%} / {sel['risk_at_50pct_coverage']:.1%}")
    passed &= alo <= sel["aurc"] <= ahi and abs(sel["risk_at_full_coverage"] - (1 - cal["accuracy"])) < 1e-9

    # Conformal prediction
    # Always the CLEAN calibration split, also for corrupted test images.
    cal_name = f"{acc['model']}_{acc['precision']}_{acc['settings']['dataset']}_conformal_calibration.json"
    cal_path = Path("results/accuracy") / cal_name
    if not cal_path.exists():
        print(f"  conformal: skipped, no {cal_path.name} (run 03 with --split conformal_calibration)")
        continue
    cal_acc, cal_arrays = load(cal_path)
    cal_probs = softmax(cal_arrays["logits"])
    threshold = conformal_threshold(cal_probs, cal_arrays["labels"], TARGET_COVERAGE)
    conf = evaluate_sets(softmax(arrays["logits"]), arrays["labels"], threshold)
    # Sanity: on the calibration images themselves coverage must reach the target by construction.
    on_cal = evaluate_sets(cal_probs, cal_arrays["labels"], threshold, n_resamples=1)["coverage"]

    settings = {**base_settings, "method": "LAC (threshold on 1 - probability of the true class)",
                "target_coverage": TARGET_COVERAGE, "threshold": threshold,
                "min_class_probability_in_set": 1 - threshold,
                "calibration_result": cal_path.name, "calibration_split": "conformal_calibration",
                "calibration_images": cal_acc["settings"]["n_images"],
                "calibration_arrays_sha256": cal_acc["raw_arrays"]["sha256"]}
    counts = conf.pop("set_size_counts")
    record = make_record("conformal", acc["model"], acc["precision"],
                         {"settings": settings, "metrics": conf, "raw": {"set_size_counts": counts}}, machine)
    save_record(record, OUT / f"{stem}_conformal.json")
    clo, chi = conf["coverage_ci95"]
    print(f"  conformal (target {TARGET_COVERAGE:.0%}): classes with probability >= "
          f"{1 - threshold:.4f} go in the set")
    print(f"    coverage {conf['coverage']:.2%} ({clo:.2%}-{chi:.2%}); "
          f"mean set size {conf['mean_set_size']:.2f}, median {conf['median_set_size']:.0f}")
    print(f"    single-class sets {conf['share_single_class']:.1%}, empty sets {conf['share_empty']:.1%}")
    passed &= on_cal >= TARGET_COVERAGE and sum(counts.values()) == len(arrays["labels"])
    passed &= not np.intersect1d(cal_arrays["positions"], arrays["positions"]).size  # never the same images

print("\nPASS" if passed else "\nFAIL")
sys.exit(0 if passed else 1)
