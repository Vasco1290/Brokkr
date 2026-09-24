"""Make FP16 and INT8 versions of an exported FP32 model.

Usage:  python scripts/04_quantize.py [--model NAME] [--calibration-images 512]
Needs:  models/<model>_fp32.onnx and data/imagenet-1k/
Writes: models/<model>_fp16.onnx, models/<model>_int8.onnx, and a .json record for each

INT8 calibration images come from the ImageNet validation set but are chosen so they never
overlap the fixed 10,000-image test subset used by scripts/03_evaluate_accuracy.py (seed 0).
They DO fall inside the full 50,000 set, which a --n 0 accuracy run must disclose.

Sanity check (not a result): on 256 test images, how often does each smaller model pick the
same top class as FP32? Below 90% prints a WARNING - a real accuracy loss worth studying.
Below 20% is a FAIL - the conversion is broken (1000 classes, so random guessing is ~0.1%).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from brokkr.accuracy import open_image, preprocess
from brokkr.benchmark import make_session
from brokkr.datasets import (DATASETS, choose_calibration, choose_subset, count_images,
                             parquet_files, read_parquet_images)
from brokkr.export import MODELS, file_info
from brokkr.fingerprint import machine_fingerprint
from brokkr.quantize import INT8_SETTINGS, to_fp16, to_int8

DATASET = "imagenet-1k-val"
TEST_SUBSET = {"n": 10_000, "seed": 0}  # must match scripts/03_evaluate_accuracy.py defaults
WARN_AGREEMENT = 0.90
FAIL_AGREEMENT = 0.20

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--calibration-images", type=int, default=512)
parser.add_argument("--calibration-seed", type=int, default=1)
args = parser.parse_args()

fp32_path = Path("models") / f"{args.model}_fp32.onnx"
if not fp32_path.exists():
    sys.exit(f"FAIL: {fp32_path} not found. Run scripts/01_export_model.py first.")


def batches_of(samples, size=32):
    batch = []
    for image, _ in samples:
        batch.append(preprocess(open_image(image)))
        if len(batch) == size:
            yield np.stack(batch)
            batch = []
    if batch:
        yield np.stack(batch)


files = parquet_files(DATASET)
total = count_images(files)
test_positions = choose_subset(total, TEST_SUBSET["n"], seed=TEST_SUBSET["seed"])
calib_positions = choose_calibration(total, args.calibration_images, exclude=test_positions,
                                     seed=args.calibration_seed)
assert not set(calib_positions) & set(test_positions), "calibration overlaps test images"

paths = {"fp16": Path("models") / f"{args.model}_fp16.onnx",
         "int8": Path("models") / f"{args.model}_int8.onnx"}
print("Converting to FP16...")
to_fp16(fp32_path, paths["fp16"])
print(f"Quantizing to INT8 with {len(calib_positions)} calibration images...")
to_int8(fp32_path, paths["int8"], batches_of(read_parquet_images(files, calib_positions)))

settings = {
    "fp16": {"method": "onnxconverter-common float16, inputs/outputs kept FP32"},
    "int8": {**INT8_SETTINGS, "calibration": {
        "dataset": DATASET, "dataset_licence": DATASETS[DATASET]["licence"],
        "n_images": len(calib_positions), "seed": args.calibration_seed,
        "excludes": f"the {TEST_SUBSET['n']}-image test subset (seed {TEST_SUBSET['seed']})"}},
}

# Sanity check: compare top-1 choices with FP32 on 256 test images.
check_images = np.concatenate(list(batches_of(read_parquet_images(files, test_positions[:256]))))


def top1(path):
    session = make_session(path, num_threads=4)
    return np.concatenate([session.run(None, {"images": check_images[i:i + 32]})[0].argmax(1)
                           for i in range(0, len(check_images), 32)])


reference = top1(fp32_path)
fp32_size = file_info(fp32_path)["size_bytes"]
machine = machine_fingerprint()
passed = True
warnings = []
print(f"\n{'precision':>9} {'size MB':>8} {'vs FP32':>8} {'top-1 agreement':>16}")
print(f"{'fp32':>9} {fp32_size / 1e6:>8.1f} {'1.00x':>8} {'-':>16}")
for precision, path in paths.items():
    agreement = float(np.mean(top1(path) == reference))
    info = file_info(path)
    record = {
        "model": args.model, "precision": precision, "derived_from": str(fp32_path),
        "licence": MODELS[args.model]["licence"], "settings": settings[precision],
        "file": {"path": str(path), **info},
        "sanity_check": {"n_images": len(check_images), "top1_agreement_with_fp32": agreement},
        "machine": machine,
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(f"{precision:>9} {info['size_bytes'] / 1e6:>8.1f} {fp32_size / info['size_bytes']:>7.2f}x "
          f"{agreement:>16.1%}")
    if agreement < WARN_AGREEMENT:
        warnings.append(f"WARNING: {precision} agrees with FP32 on only {agreement:.1%} "
                        f"(below {WARN_AGREEMENT:.0%}) - expect a real accuracy drop")
    passed &= agreement >= FAIL_AGREEMENT

print("", *warnings, sep="\n")
print("PASS" if passed else f"FAIL: a model agrees with FP32 on fewer than {FAIL_AGREEMENT:.0%}")
sys.exit(0 if passed else 1)
