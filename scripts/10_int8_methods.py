"""Build INT8 models with different calibration methods, for choosing the best one (task 3.2).

Usage:  python scripts/10_int8_methods.py [--model NAME] [--methods percentile99.99 ...]
Needs:  models/<model>_fp32.onnx, models/<model>_int8.onnx (the default MinMax INT8 from
        scripts/04_quantize.py) and data/imagenet-1k/
Writes: models/<model>_int8_<method>.onnx and a .json record for each method

Every candidate uses the same 512 "int8_calibration" images, per-channel int8 weights and per-tensor
uint8 activations; only the calibration method differs (brokkr.quantize.INT8_METHODS). The images
are fed in groups of 128 (4 batches of 32), in the split's fixed order, because the histogram
methods would otherwise need about 22 GB of memory.

The MinMax candidate is the existing default INT8 model. This script rebuilds MinMax with the same
grouping as the other methods and checks it gives exactly the same outputs, so grouping cannot
explain any difference between methods.

Next: measure each candidate on the clean tuning split with
    python scripts/03_evaluate_accuracy.py --precision int8_<method> --split tuning
then choose with scripts/11_choose_int8.py. The test split is not used here at all.

Sanity check (not a result): top-1 agreement with FP32 on 256 tuning images.
Below 20% is a FAIL (the conversion is broken).
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from brokkr.accuracy import open_image, preprocess
from brokkr.benchmark import make_session
from brokkr.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr.export import MODELS, file_info
from brokkr.fingerprint import machine_fingerprint
from brokkr.quantize import CALIBRATION_BATCH as BATCH
from brokkr.quantize import CALIBRATION_GROUP_BATCHES as GROUP_BATCHES
from brokkr.quantize import INT8_METHODS, INT8_SETTINGS, to_int8

DATASET = "imagenet-1k-val"
FAIL_AGREEMENT = 0.20

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--methods", nargs="+", default=[m for m in INT8_METHODS if m != "minmax"],
                    choices=[m for m in INT8_METHODS if m != "minmax"])
args = parser.parse_args()

fp32_path = Path("models") / f"{args.model}_fp32.onnx"
default_int8 = Path("models") / f"{args.model}_int8.onnx"
for path in (fp32_path, default_int8):
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run scripts/01_export_model.py and 04_quantize.py first.")


def load_batches(files, positions):
    images = np.stack([preprocess(open_image(image)) for image, _ in read_parquet_images(files, positions)])
    return [images[i:i + BATCH] for i in range(0, len(images), BATCH)]


def outputs(path, batches):
    session = make_session(path, num_threads=4)
    return np.concatenate([session.run(None, {"images": b})[0] for b in batches])


files = parquet_files(DATASET)
splits = make_splits(count_images(files))
print(f"Loading {len(splits['int8_calibration'])} calibration images and 256 tuning images...")
calibration = load_batches(files, splits["int8_calibration"])
check = load_batches(files, splits["tuning"][:256])
reference = outputs(fp32_path, check).argmax(1)

# 1. Grouping check: MinMax built in groups must equal the existing default INT8 model exactly.
with tempfile.TemporaryDirectory() as tmp:
    grouped_minmax = to_int8(fp32_path, Path(tmp) / "minmax_grouped.onnx", calibration,
                             method="minmax", group_batches=GROUP_BATCHES)
    grouping_ok = np.array_equal(outputs(grouped_minmax, check), outputs(default_int8, check))
print(f"MinMax built in groups gives exactly the default INT8 outputs: {grouping_ok}")

# 2. Build each candidate and write its record.
machine = machine_fingerprint()
passed = grouping_ok
print(f"\n{'candidate':>22} {'size MB':>8} {'top-1 agreement with FP32':>26}")
for method in args.methods:
    precision = f"int8_{method}"
    path = Path("models") / f"{args.model}_{precision}.onnx"
    to_int8(fp32_path, path, calibration, method=method, group_batches=GROUP_BATCHES)
    agreement = float(np.mean(outputs(path, check).argmax(1) == reference))
    info = file_info(path)
    record = {
        "model": args.model, "precision": precision, "derived_from": str(fp32_path),
        "licence": MODELS[args.model]["licence"],
        "settings": {**INT8_SETTINGS, "calibration_method": method,
                     "calibration_method_detail": INT8_METHODS[method]["description"],
                     "calibration": {"dataset": DATASET, "dataset_licence": DATASETS[DATASET]["licence"],
                                     "split": "int8_calibration", "n_images": len(splits["int8_calibration"]),
                                     "group_images": GROUP_BATCHES * BATCH,
                                     "excludes": "all other splits, including test"}},
        "file": {"path": str(path), **info},
        "sanity_check": {"split": "tuning", "n_images": len(reference),
                         "top1_agreement_with_fp32": agreement},
        "machine": machine,
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(f"{precision:>22} {info['size_bytes'] / 1e6:>8.1f} {agreement:>26.1%}", flush=True)
    passed &= agreement >= FAIL_AGREEMENT

print("\nPASS" if passed else "\nFAIL: grouping changed MinMax, or a model agrees with FP32 below 20%")
sys.exit(0 if passed else 1)
