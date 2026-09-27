"""Task 4.1: reliability numbers for every breadth-sweep result, from its saved scores (no model is re-run).

Usage:  python scripts/31_breadth_reliability.py                        (the 4.1 test-split results)
        python scripts/31_breadth_reliability.py --results <dry-run folder> --split tuning --out <folder>
Needs:  schema-2 accuracy records with saved scores from scripts/28_breadth_sweep.py (--results), and
        each model and precision's clean conformal_calibration record (--calibration; the full run
        writes them to results/breadth)
Writes: <out>/<source name>_calibration.json, _conformal.json, _selective.json (schema-2 records;
        default out: results/breadth_reliability)

The metrics are Stage 3's, computed by the same tested functions as scripts/07_reliability.py:
- calibration: ECE (15 equal-width bins) with a bootstrap 95% interval (brokkr.shift.reliability);
- conformal: the 90% threshold from the SAME model and precision's clean conformal_calibration images
  (5,000 in the full run), then coverage AND average set size on the evaluated images, each with a
  bootstrap 95% interval (brokkr.shift.conformal); coverage is never reported without set size;
- selective prediction: AURC and E-AURC with bootstrap 95% intervals (brokkr.shift.selective).
Bootstrap: 1,000 resamples, seed 0. Nothing here is judged; H18-H22 use top-1 only (scripts/32).

Checks (PASS / FAIL at the end): each record's score file matches its checksum; ECE and its interval
lie in [0, 1]; the accuracy recomputed here equals the record's top-1; AURC lies in its interval and
the error rate at full coverage is 1 - accuracy; the threshold reaches 90% coverage on its own
calibration images; the set-size counts add up to the image count; the calibration and evaluated
images never overlap; every record passes the schema check.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from brokkr.fingerprint import machine_fingerprint
from brokkr.schema import condition_label, load_measurement, make_measurement, metric, save_measurement
from brokkr.shift.conformal import conformal_threshold, evaluate_sets
from brokkr.shift.reliability import calibration, confidence_and_correct, softmax
from brokkr.shift.selective import selective_prediction

TARGET_COVERAGE, RESAMPLES, SEED = 0.90, 1000, 0

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results/breadth", help="folder with the accuracy records")
parser.add_argument("--calibration", default="results/breadth",
                    help="folder with the clean conformal_calibration records")
parser.add_argument("--split", default="test", help="which split's records to compute for")
parser.add_argument("--out", default="results/breadth_reliability")
args = parser.parse_args()
out_dir = Path(args.out)
machine = machine_fingerprint()


def accuracy_records(folder, split: str) -> list:
    """Paths of the schema-2 accuracy records with saved scores on `split`, sorted by name."""
    found = []
    for path in sorted(Path(folder).glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if (r.get("schema_version") == 2 and r["kind"] == "accuracy" and r["data"]["split"] == split
                and r["arrays"]):
            found.append(path)
    return found


def save(kind: str, source: Path, acc: dict, metrics: dict, settings: dict, raw: dict,
         also_derived_from=()) -> None:
    """One schema-2 reliability record, computed from the accuracy record `acc` saved at `source`."""
    record = make_measurement(
        kind, acc["model"], acc["precision"], acc["runtime"], "laptop", machine, acc["data"],
        acc["condition"], metrics,
        {"script": "scripts/31_breadth_reliability.py", "bootstrap_resamples": RESAMPLES, "seed": SEED,
         **settings},
        None, [{"file": source.as_posix(), "arrays_sha256": acc["arrays"]["sha256"]}, *also_derived_from])
    record["raw"] = raw
    save_measurement(record, out_dir / f"{source.stem}_{kind}.json")


def interval(values: list, digits: int) -> str:
    return f"({values[0]:.{digits}f} to {values[1]:.{digits}f})"


sources = accuracy_records(args.results, args.split)
if not sources:
    sys.exit(f"FAIL: no {args.split}-split accuracy records with scores in {args.results}")
calibration_records = {}
for p in accuracy_records(args.calibration, "conformal_calibration"):
    r = json.loads(p.read_text(encoding="utf-8"))
    if r["condition"]["corruption"] == "clean":
        calibration_records[(r["model"]["name"], r["precision"])] = p

thresholds = {}  # (model, precision) -> (threshold, calibration path, record, positions, coverage there)
passed = True
print(f"{len(sources)} {args.split}-split records in {args.results}; out: {out_dir}")
for path in sources:
    acc, arrays = load_measurement(path)
    model, precision = acc["model"]["name"], acc["precision"]
    logits, labels = arrays["logits"], arrays["labels"]

    # Calibration
    cal = calibration(logits, labels)
    save("calibration", path, acc,
         {"ece": metric(cal["ece"], cal["ece_ci95"]), "mean_confidence": metric(cal["mean_confidence"]),
          "accuracy": metric(cal["accuracy"]), "overconfidence": metric(cal["overconfidence"])},
         {"n_bins": cal["n_bins"]}, {"bins": cal["bins"]})
    # ECE is biased upwards, so near zero its interval can sit entirely above it
    # (brokkr/shift/reliability.py): the check is that everything lies in [0, 1].
    ok = 0 <= cal["ece"] <= 1 and 0 <= cal["ece_ci95"][0] <= cal["ece_ci95"][1] <= 1
    ok &= abs(cal["accuracy"] - acc["metrics"]["top1"]["value"]) < 1e-9

    # Selective prediction
    confidence, correct = confidence_and_correct(logits, labels)
    sel = selective_prediction(confidence, correct)
    curve = sel.pop("curve")
    risks = ("risk_at_full_coverage", "risk_at_80pct_coverage", "risk_at_50pct_coverage")
    save("selective", path, acc,
         {"e_aurc": metric(sel["e_aurc"], sel["e_aurc_ci95"]), "aurc": metric(sel["aurc"], sel["aurc_ci95"]),
          "optimal_aurc": metric(sel["optimal_aurc"]), **{k: metric(sel[k]) for k in risks}},
         {"confidence": "probability of the top answer"}, {"risk_coverage_curve": curve})
    ok &= sel["aurc_ci95"][0] <= sel["aurc"] <= sel["aurc_ci95"][1]
    ok &= abs(sel["risk_at_full_coverage"] - (1 - cal["accuracy"])) < 1e-9

    # Conformal prediction: always the clean conformal_calibration images of the same model and precision.
    if (model, precision) not in thresholds:
        cal_path = calibration_records.get((model, precision))
        if cal_path is None:
            sys.exit(f"FAIL: no clean conformal_calibration record for {model} {precision} "
                     f"in {args.calibration}")
        cal_acc, cal_arrays = load_measurement(cal_path)
        cal_probs = softmax(cal_arrays["logits"])
        q = conformal_threshold(cal_probs, cal_arrays["labels"], TARGET_COVERAGE)
        on_own = evaluate_sets(cal_probs, cal_arrays["labels"], q, n_resamples=1)["coverage"]
        thresholds[(model, precision)] = (q, cal_path, cal_acc, cal_arrays["positions"], on_own)
    q, cal_path, cal_acc, cal_positions, on_own = thresholds[(model, precision)]
    conf = evaluate_sets(softmax(logits), labels, q)
    counts = conf.pop("set_size_counts")
    save("conformal", path, acc,
         {"coverage": metric(conf["coverage"], conf["coverage_ci95"]),
          "mean_set_size": metric(conf["mean_set_size"], conf["mean_set_size_ci95"]),
          **{k: metric(conf[k]) for k in ("median_set_size", "share_single_class", "share_empty")}},
         {"method": "LAC (threshold on 1 - probability of the true class)",
          "target_coverage": TARGET_COVERAGE,
          "threshold": q, "min_class_probability_in_set": 1 - q, "calibration_split": "conformal_calibration",
          "calibration_images": cal_acc["data"]["n_images"], "coverage_on_calibration_images": on_own},
         {"set_size_counts": counts},
         [{"file": cal_path.as_posix(), "arrays_sha256": cal_acc["arrays"]["sha256"]}])
    ok &= on_own >= TARGET_COVERAGE and sum(counts.values()) == len(labels)
    ok &= not np.intersect1d(cal_positions, arrays["positions"]).size  # never the same images

    passed &= ok
    print(f"{'ok  ' if ok else 'FAIL'} {model:<19} {precision:<21} {condition_label(acc['condition']):<30} "
          f"n={len(labels):,}  ECE {cal['ece']:.4f} {interval(cal['ece_ci95'], 4)}  "
          f"coverage {conf['coverage']:.4f} {interval(conf['coverage_ci95'], 4)} "
          f"set {conf['mean_set_size']:.2f} {interval(conf['mean_set_size_ci95'], 2)}  "
          f"E-AURC {sel['e_aurc']:.4f} {interval(sel['e_aurc_ci95'], 4)}")

print(f"\nConformal thresholds (clean conformal_calibration, target {TARGET_COVERAGE:.0%}):")
for (model, precision), (q, _, cal_acc, _, on_own) in sorted(thresholds.items()):
    print(f"  {model:<19} {precision:<21} threshold {q:.6f} on {cal_acc['data']['n_images']:,} images, "
          f"coverage there {on_own:.4f}")
print(f"\n{'PASS' if passed else 'FAIL'}: {3 * len(sources)} records from {len(sources)} results "
      f"({len(thresholds)} conformal thresholds)")
sys.exit(0 if passed else 1)
