"""ImageNetV2 (task 3.9): top-1 and clean-tuned conformal coverage WITH set size, FP32 and best INT8.

Usage:  python scripts/21_imagenetv2_summary.py [--model NAME]
Needs:  results/accuracy/<model>_<p>_imagenetv2-matched-frequency_all.json for FP32 and best INT8
        (scripts/03_evaluate_accuracy.py --dataset imagenetv2-matched-frequency --split all),
        results/choices/<model>_int8_method.json and results/choices/<model>_shift_aware.json
Writes: results/final/<model>_imagenetv2_summary.json

FP32 is what H17 is about (judged by scripts/18_judge_stage3.py). Best INT8 is an extra analysis, not a
prediction (dated 3.9 note in docs/hypotheses_stage3.md). Each model uses its own conformal threshold
tuned on CLEAN ImageNet conformal_calibration images (task 3.6), so this asks whether the 90% promise
survives new photos. ImageNet test-split numbers are shown next to them for comparison.
"""

import argparse
import json
from pathlib import Path

from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.results import load_arrays, make_record, save_record
from brokkr_edge.shift.conformal import evaluate_sets
from brokkr_edge.shift.reliability import softmax

V2 = "imagenetv2-matched-frequency"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
args = parser.parse_args()
m = args.model

best = json.loads((Path("results/choices") / f"{m}_int8_method.json").read_text())["metrics"]["chosen"]
shift = json.loads((Path("results/choices") / f"{m}_shift_aware.json").read_text())["metrics"]

rows = {}
for precision, label in (("fp32", "FP32 (H17)"), (best, "best INT8 (extra analysis)")):
    threshold = shift["robust_conformal"][precision]["clean_only_threshold"]
    row = {}
    sources = {V2: Path("results/accuracy") / f"{m}_{precision}_{V2}_all.json",
               "imagenet test": Path("results/sweep") / f"{m}_{precision}_imagenet-1k-val_test_clean_s0.json"}
    for dataset, path in sources.items():
        record, arrays = json.loads(path.read_text()), load_arrays(path)
        sets = evaluate_sets(softmax(arrays["logits"]), arrays["labels"], threshold)
        row[dataset] = {"n_images": len(arrays["labels"]), "top1": record["metrics"]["top1"],
                        "top1_ci95": record["metrics"]["top1_ci95"], **sets}
    rows[precision] = row
    print(f"\n{label}, clean-tuned threshold {threshold:.6f}")
    for dataset, r in row.items():
        print(f"    {dataset:<22} top-1 {r['top1']:.2%} ({r['top1_ci95'][0]:.2%}-{r['top1_ci95'][1]:.2%}); "
              f"coverage {r['coverage']:.2%} ({r['coverage_ci95'][0]:.2%}-{r['coverage_ci95'][1]:.2%}), "
              f"average set size {r['mean_set_size']:.2f}")

record = make_record("imagenetv2_summary", m, "all", {
    "settings": {"dataset": V2, "dataset_citation": "Recht et al., ICML 2019",
                 "thresholds": "each model's clean-tuned conformal threshold (task 3.6)",
                 "best_int8": "extra analysis, not a prediction"},
    "metrics": rows,
}, machine_fingerprint())
out = save_record(record, Path("results/final") / f"{m}_imagenetv2_summary.json")
print(f"\nSaved to {out}")
