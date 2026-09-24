"""Measure top-1/top-5 accuracy of an exported ONNX model on the ImageNet validation set.

Usage:  python scripts/03_evaluate_accuracy.py [--model NAME] [--precision fp32]
                                                [--split test|conformal_calibration|tuning|all]
        test:                  the fixed 10,000-image test split (brokkr.datasets.make_splits)
        conformal_calibration: 5,000 images for tuning conformal prediction sets
        tuning:                5,000 images held back for Stage 3
        all:                   all 50,000 images (includes the INT8 calibration images; FP32 check only)
Needs:  models/<model>_<precision>.onnx and data/imagenet-1k/ (see README)
Writes: results/accuracy/<model>_<precision>_imagenet-1k-val_<split>.json, plus a .npz file with
        every class score (logit) for every image, so reliability metrics can be computed later

Correctness check (FP32 only): torchvision publishes its own top-1 accuracy for these FP32
weights on this same dataset. If our pipeline is right, that published number should fall
inside our 95% confidence interval. (Even a correct pipeline misses about 1 time in 20.)
For FP16/INT8 a lower accuracy is a finding, not an error, so instead the script prints the
change versus our own FP32 result on the same images, if that result exists.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from brokkr.accuracy import evaluate
from brokkr.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr.export import MODELS
from brokkr.fingerprint import machine_fingerprint
from brokkr.results import make_record, save_arrays, save_record

DATASET = "imagenet-1k-val"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--precision", default="fp32")
parser.add_argument("--split", default="test",
                    choices=["test", "conformal_calibration", "tuning", "all"])
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--seed", type=int, default=0, help="seed for the bootstrap intervals")
args = parser.parse_args()

onnx_path = Path("models") / f"{args.model}_{args.precision}.onnx"
if not onnx_path.exists():
    sys.exit(f"FAIL: {onnx_path} not found. Run scripts/01_export_model.py first.")

files = parquet_files(DATASET)
total = count_images(files)
positions = np.arange(total) if args.split == "all" else make_splits(total)[args.split]
print(f"Evaluating {onnx_path} on {len(positions)} of {total} images ({DATASET})...")

start = time.time()
result, logits = evaluate(onnx_path, read_parquet_images(files, positions),
                  num_threads=args.threads, seed=args.seed)
result["settings"].update({"dataset": DATASET, "dataset_total_images": total,
                           "dataset_licence": DATASETS[DATASET]["licence"],
                           "split": args.split})

# Published number from torchvision, kept separate from our own measurements.
reported = MODELS[args.model]["weights"].meta["_metrics"]["ImageNet-1K"]["acc@1"] / 100
result["reference"] = {"top1_reported_by_torchvision": reported,
                       "note": "Published by torchvision for the PyTorch model on all 50,000 images."}

record = make_record("accuracy", args.model, args.precision, result, machine_fingerprint())
out = Path("results/accuracy") / f"{args.model}_{args.precision}_{DATASET}_{args.split}.json"
save_arrays(record, out, logits=logits, labels=np.array(result["raw"]["labels"]),
            positions=positions)
save_record(record, out)

m = result["metrics"]
lo, hi = m["top1_ci95"]
print(f"Done in {time.time() - start:.0f} s. Saved to {out}\n")
print(f"Top-1: {m['top1']:.2%}  (95% CI {lo:.2%} - {hi:.2%})")
print(f"Top-5: {m['top5']:.2%}  (95% CI {m['top5_ci95'][0]:.2%} - {m['top5_ci95'][1]:.2%})")
if args.precision == "fp32" and args.split in ("test", "all"):
    print(f"torchvision's published top-1: {reported:.2%}")
    passed = lo <= reported <= hi
    print("PASS: published number is inside our confidence interval" if passed else
          "FAIL: published number is outside our confidence interval - check preprocessing/labels")
else:
    fp32_file = out.with_name(out.name.replace(f"_{args.precision}_", "_fp32_"))
    if fp32_file.exists():
        fp32_top1 = json.loads(fp32_file.read_text())["metrics"]["top1"]
        print(f"Change vs our FP32 on the same images: {(m['top1'] - fp32_top1) * 100:+.2f} points "
              f"(FP32 {fp32_top1:.2%})")
    else:
        print(f"(No FP32 result at {fp32_file} to compare with.)")
    # Basic sanity: every requested image was evaluated and the metrics make sense.
    passed = result["settings"]["n_images"] == len(positions) and 0 <= m["top1"] <= m["top5"] <= 1
    print("PASS" if passed else "FAIL: evaluation incomplete or metrics inconsistent")
sys.exit(0 if passed else 1)
