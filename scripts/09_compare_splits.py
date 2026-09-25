"""Compare FP32 clean accuracy across the data splits, and show how the splits were drawn.

Usage:  python scripts/09_compare_splits.py [--model NAME] [--precision fp32]
Needs:  results/accuracy/<model>_<precision>_imagenet-1k-val_<split>.json (+ .npz) for the test,
        tuning and conformal_calibration splits (scripts/03_evaluate_accuracy.py --split ...)
Prints: top-1 per split, each split's difference from the test split with a 95% bootstrap interval
        (the splits are different images, so each side is resampled on its own), and how many
        images each class has in each split.

How the splits are drawn (brokkr.datasets.make_splits): at random with fixed seeds, NOT stratified
by class, so the number of images per class varies from split to split.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from brokkr.accuracy import unpaired_bootstrap_diff
from brokkr.datasets import SPLIT_SIZES
from brokkr.results import load_arrays

DATASET = "imagenet-1k-val"
SPLITS = ["test", "tuning", "conformal_calibration"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--precision", default="fp32")
args = parser.parse_args()

data = {}
for split in SPLITS:
    path = Path("results/accuracy") / f"{args.model}_{args.precision}_{DATASET}_{split}.json"
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run scripts/03_evaluate_accuracy.py --split {split} first.")
    arrays = load_arrays(path)  # checks the .npz checksum
    top1 = np.argsort(-arrays["logits"], axis=1, kind="stable")[:, 0]  # same tie rule as everywhere
    data[split] = {"correct": (top1 == arrays["labels"]).astype(float), "labels": arrays["labels"],
                   "positions": arrays["positions"]}

print(f"{args.model} {args.precision}, clean images, top-1 accuracy by split")
print(f"{'split':<24}{'images':>7}{'top-1':>9}   difference vs test (95% CI, unpaired bootstrap)")
test = data["test"]["correct"]
for split in SPLITS:
    c = data[split]["correct"]
    line = f"{split:<24}{len(c):>7,}{c.mean():>9.2%}"
    if split != "test":
        diff, low, high = unpaired_bootstrap_diff(test, c)
        line += f"   {diff * 100:+.2f} points ({low * 100:+.2f} to {high * 100:+.2f})"
    print(line)

print("\nImages per class (1,000 classes; drawn at random, not stratified)")
print(f"{'split':<24}{'expected':>9}{'min':>6}{'median':>8}{'max':>6}{'classes with 0':>16}")
for split in SPLITS:
    counts = np.bincount(data[split]["labels"], minlength=1000)
    print(f"{split:<24}{len(data[split]['labels']) / 1000:>9.1f}{counts.min():>6}"
          f"{np.median(counts):>8.0f}{counts.max():>6}{(counts == 0).sum():>16}")

# Checks: every split is complete and no image is in two splits.
sizes_ok = all(len(data[s]["correct"]) == SPLIT_SIZES[s] for s in SPLITS)
overlap = sum(len(np.intersect1d(data[a]["positions"], data[b]["positions"]))
              for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:])
print(f"\nSplit sizes as defined: {sizes_ok}. Images shared between splits: {overlap}")
passed = sizes_ok and overlap == 0
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
