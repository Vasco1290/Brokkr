"""Measure top-1/top-5 accuracy of an exported ONNX model on the ImageNet validation set.

Usage:  python scripts/03_evaluate_accuracy.py [--model NAME] [--precision fp32] [--n 10000 | --n 0]
        --n 0 means all 50,000 images.
Needs:  models/<model>_<precision>.onnx and data/imagenet-1k/ (see README)
Writes: results/accuracy/<model>_<precision>_imagenet-1k-val_<n>.json

Correctness check: torchvision publishes its own top-1 accuracy for these weights on this
same dataset. If our pipeline is right, that published number should fall inside our 95%
confidence interval. (Even a correct pipeline misses about 1 time in 20 by chance.)
"""

import argparse
import sys
import time
from pathlib import Path

from brokkr.accuracy import evaluate
from brokkr.datasets import DATASETS, choose_subset, count_images, parquet_files, read_parquet_images
from brokkr.export import MODELS
from brokkr.fingerprint import machine_fingerprint
from brokkr.results import make_record, save_record

DATASET = "imagenet-1k-val"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--precision", default="fp32")
parser.add_argument("--n", type=int, default=10_000, help="number of images; 0 = all")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

onnx_path = Path("models") / f"{args.model}_{args.precision}.onnx"
if not onnx_path.exists():
    sys.exit(f"FAIL: {onnx_path} not found. Run scripts/01_export_model.py first.")

files = parquet_files(DATASET)
total = count_images(files)
positions = choose_subset(total, args.n or None, seed=args.seed)
print(f"Evaluating {onnx_path} on {len(positions)} of {total} images ({DATASET})...")

start = time.time()
result = evaluate(onnx_path, read_parquet_images(files, positions),
                  num_threads=args.threads, seed=args.seed)
result["settings"].update({"dataset": DATASET, "dataset_total_images": total,
                           "dataset_licence": DATASETS[DATASET]["licence"],
                           "subset_seed": args.seed})

# Published number from torchvision, kept separate from our own measurements.
reported = MODELS[args.model]["weights"].meta["_metrics"]["ImageNet-1K"]["acc@1"] / 100
result["reference"] = {"top1_reported_by_torchvision": reported,
                       "note": "Published by torchvision for the PyTorch model on all 50,000 images."}

record = make_record("accuracy", args.model, args.precision, result, machine_fingerprint())
out = Path("results/accuracy") / f"{args.model}_{args.precision}_{DATASET}_{len(positions)}.json"
save_record(record, out)

m = result["metrics"]
lo, hi = m["top1_ci95"]
print(f"Done in {time.time() - start:.0f} s. Saved to {out}\n")
print(f"Top-1: {m['top1']:.2%}  (95% CI {lo:.2%} - {hi:.2%})")
print(f"Top-5: {m['top5']:.2%}  (95% CI {m['top5_ci95'][0]:.2%} - {m['top5_ci95'][1]:.2%})")
print(f"torchvision's published top-1: {reported:.2%}")

passed = lo <= reported <= hi
print("PASS: published number is inside our confidence interval" if passed else
      "FAIL: published number is outside our confidence interval - check preprocessing/labels")
sys.exit(0 if passed else 1)
