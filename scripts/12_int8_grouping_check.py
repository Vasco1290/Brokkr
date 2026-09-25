"""Check how the calibration group size affects a histogram-based INT8 model (before task 3.3).

Usage:  python scripts/12_int8_grouping_check.py [--model NAME] [--method percentile99.99]
Needs:  models/<model>_int8_<method>.onnx (scripts/10_int8_methods.py), its clean tuning result
        (scripts/03_evaluate_accuracy.py --precision int8_<method> --split tuning), the tuning image
        cache (scripts/08_corruption_sweep.py --split tuning) and data/imagenet-1k/
Writes: models/<model>_int8_<method>_group64.onnx (+ .json record) and
        results/checks/<model>_int8_<method>_grouping.json (+ .npz with the group-64 tuning logits)

(a) Repeatability: rebuild with the standard group size (128 images) and require the SAME model:
    every stored number equal, and identical scores on all 5,000 tuning images. FAIL otherwise.
(b) Sensitivity: build once with groups of 64 images and report its clean tuning top-1 and the
    paired difference from the group-128 model. This is information only; the group size stays 128
    (docs/hypotheses_stage3.md).

Tuning scores are computed from the cached tuning pictures (normalize(crop)), the path that the
sweep's clean check showed gives exactly the same scores as scripts/03.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from brokkr.accuracy import accuracy_from_logits, normalize, open_image, paired_bootstrap_diff, preprocess
from brokkr.benchmark import make_session
from brokkr.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr.export import MODELS, file_info
from brokkr.fingerprint import machine_fingerprint
from brokkr.quantize import CALIBRATION_BATCH as BATCH
from brokkr.quantize import CALIBRATION_GROUP_BATCHES as GROUP_BATCHES
from brokkr.quantize import INT8_METHODS, INT8_SETTINGS, to_int8
from brokkr.results import load_arrays, make_record, save_arrays, save_record

DATASET = "imagenet-1k-val"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--method", default="percentile99.99", choices=[m for m in INT8_METHODS if m != "minmax"])
args = parser.parse_args()

fp32_path = Path("models") / f"{args.model}_fp32.onnx"
standard = Path("models") / f"{args.model}_int8_{args.method}.onnx"
standard_result = Path("results/accuracy") / f"{args.model}_int8_{args.method}_{DATASET}_tuning.json"
crops_path = Path("data/cache") / f"{DATASET}_tuning_crops.npy"
for path in (fp32_path, standard, standard_result, crops_path):
    if not path.exists():
        sys.exit(f"FAIL: {path} not found (see Needs in this script's docstring)")

files = parquet_files(DATASET)
splits = make_splits(count_images(files))
images = np.stack([preprocess(open_image(image))
                   for image, _ in read_parquet_images(files, splits["int8_calibration"])])
calibration = [images[i:i + BATCH] for i in range(0, len(images), BATCH)]
crops = np.load(crops_path, mmap_mode="r")
reference = load_arrays(standard_result)  # the group-128 model's scores from scripts/03


def tuning_logits(path):
    session = make_session(path, num_threads=4)
    batches = (np.stack([normalize(c) for c in crops[i:i + BATCH]]) for i in range(0, len(crops), BATCH))
    return np.concatenate([session.run(None, {"images": b})[0] for b in batches]).astype(np.float32)


def stored_numbers(path):
    return {init.name: numpy_helper.to_array(init) for init in onnx.load(str(path)).graph.initializer}


# (a) Rebuild with groups of 128 and compare with the existing model.
print(f"(a) Rebuilding int8_{args.method} with groups of {GROUP_BATCHES * BATCH} images...")
with tempfile.TemporaryDirectory() as tmp:
    rebuilt = to_int8(fp32_path, Path(tmp) / "rebuilt.onnx", calibration, method=args.method,
                      group_batches=GROUP_BATCHES)
    old, new = stored_numbers(standard), stored_numbers(rebuilt)
    same_numbers = old.keys() == new.keys() and all(np.array_equal(old[k], new[k]) for k in old)
    same_scores = np.array_equal(tuning_logits(rebuilt), reference["logits"])
print(f"    every stored number identical: {same_numbers}")
print(f"    identical scores on all {len(crops):,} tuning images: {same_scores}")

# (b) Build once with groups of 64 and compare clean tuning top-1 (paired, same images).
precision = f"int8_{args.method}_group64"
group64 = Path("models") / f"{args.model}_{precision}.onnx"
print(f"\n(b) Building {precision} with groups of {GROUP_BATCHES // 2 * BATCH} images...")
to_int8(fp32_path, group64, calibration, method=args.method, group_batches=GROUP_BATCHES // 2)
machine = machine_fingerprint()
group64.with_suffix(".json").write_text(json.dumps({
    "model": args.model, "precision": precision, "derived_from": str(fp32_path),
    "licence": MODELS[args.model]["licence"],
    "settings": {**INT8_SETTINGS, "calibration_method": args.method,
                 "calibration_method_detail": INT8_METHODS[args.method]["description"],
                 "calibration": {"dataset": DATASET, "dataset_licence": DATASETS[DATASET]["licence"],
                                 "split": "int8_calibration", "n_images": len(images),
                                 "group_images": GROUP_BATCHES // 2 * BATCH,
                                 "purpose": "group-size sensitivity check only; not a Stage 3 candidate"}},
    "file": {"path": str(group64), **file_info(group64)}, "machine": machine}, indent=2))

logits64 = tuning_logits(group64)
labels = reference["labels"]
top1 = {name: (np.argsort(-z, axis=1, kind="stable")[:, 0] == labels).astype(float)
        for name, z in (("group128", reference["logits"]), ("group64", logits64))}
diff, low, high = paired_bootstrap_diff(top1["group128"], top1["group64"])
agreement = float(np.mean(logits64.argmax(1) == reference["logits"].argmax(1)))
print(f"    clean tuning top-1: groups of 128 {top1['group128'].mean():.2%}, "
      f"groups of 64 {top1['group64'].mean():.2%}")
print(f"    64 minus 128: {diff * 100:+.2f} points (paired 95% CI {low * 100:+.2f} to {high * 100:+.2f}); "
      f"same top answer on {agreement:.1%} of images")

result = accuracy_from_logits(logits64, labels)
result["settings"] = {"dataset": DATASET, "split": "tuning", "n_images": len(labels),
                      "compared_with": standard.name, "bootstrap_resamples": 1000, "seed": 0}
result["metrics"].update({"repeat_build_identical_numbers": same_numbers,
                          "repeat_build_identical_scores": same_scores,
                          "group128_top1": float(top1["group128"].mean()),
                          "group64_minus_group128": diff, "group64_minus_group128_ci95": [low, high],
                          "top1_agreement_group64_vs_group128": agreement})
record = make_record("check", args.model, precision, result, machine)
out = Path("results/checks") / f"{args.model}_int8_{args.method}_grouping.json"
save_arrays(record, out, logits=logits64, labels=labels, positions=reference["positions"])
save_record(record, out)
print(f"Saved to {out}")

passed = same_numbers and same_scores
print("PASS: rebuilding gives the same model" if passed else "FAIL: rebuilding did not give the same model")
sys.exit(0 if passed else 1)
