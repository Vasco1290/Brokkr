"""Build best INT8 calibrated on half clean, half damaged images, leaving one corruption out (task 3.3).

Usage:  python scripts/13_int8_damaged_calibration.py [--model NAME] [--held-out fog darkness ...]
Needs:  models/<model>_fp32.onnx, results/choices/<model>_int8_method.json (scripts/11) and
        data/imagenet-1k/
Writes: models/<model>_<best>_mixed_without_<corruption>.onnx (+ .json record), one per held-out
        corruption, e.g. mobilenet_v3_large_int8_percentile99.99_mixed_without_darkness.onnx

How each model is built (docs/hypotheses_stage3.md, design rules and dated notes):
- The chosen INT8 method (from scripts/11), the same 512 int8_calibration images, per-channel
  weights, groups of 128 images: everything as for "best INT8" except what the images show.
- Exactly half of the 512 images are damaged: a random corruption from the four NOT held out, at a
  random severity 1-5 (brokkr_edge.quantize.damaged_calibration_plan, seed 5). The same images get damaged
  at the same severities for all five models. Each image's damage pattern is seeded by its dataset
  position, as in the sweep.

Safety: each model is written to a temporary file and checked before it gets its real name: it
loads; on 256 clean tuning images it gives finite scores of the right shape; its top answer agrees
with FP32 on at least 20% of them; every int8 weight tensor is per-channel. The run stops at the
first failure (or crash) and deletes the temporary file, so no partial model is ever left behind.
Models that already exist are skipped (they passed these checks when they were built).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from brokkr_edge.accuracy import normalize, open_image, resize_and_crop
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.export import MODELS, file_info
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.quantize import CALIBRATION_BATCH as BATCH
from brokkr_edge.quantize import CALIBRATION_GROUP_BATCHES as GROUP_BATCHES
from brokkr_edge.quantize import (
    INT8_METHODS,
    INT8_SETTINGS,
    damaged_calibration_plan,
    to_int8,
    weight_quantization,
)
from brokkr_edge.shift import CORRUPTIONS, corrupt

DATASET = "imagenet-1k-val"
PLAN_SEED = 5
FAIL_AGREEMENT = 0.20

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--held-out", nargs="+", default=list(CORRUPTIONS), choices=list(CORRUPTIONS))
args = parser.parse_args()

fp32_path = Path("models") / f"{args.model}_fp32.onnx"
choice_path = Path("results/choices") / f"{args.model}_int8_method.json"
for path in (fp32_path, choice_path):
    if not path.exists():
        sys.exit(f"FAIL: {path} not found (see Needs in this script's docstring)")
choice = json.loads(choice_path.read_text())["metrics"]
best, method = choice["chosen"], choice["chosen_method"]
best_path = Path("models") / f"{args.model}_{best}.onnx"
print(f"Best INT8 (task 3.2): {best} ({method})")

files = parquet_files(DATASET)
splits = make_splits(count_images(files))
positions = splits["int8_calibration"]
print(f"Loading {len(positions)} calibration images and 256 clean tuning images...")
crops = np.stack([resize_and_crop(open_image(image)) for image, _ in read_parquet_images(files, positions)])
check = np.stack([normalize(resize_and_crop(open_image(image)))
                  for image, _ in read_parquet_images(files, splits["tuning"][:256])])


def scores(path):
    session = make_session(path, num_threads=4)
    return np.concatenate([session.run(None, {"images": check[i:i + BATCH]})[0]
                           for i in range(0, len(check), BATCH)])


fp32_top1 = scores(fp32_path).argmax(1)
expected_weights = weight_quantization(best_path)  # the chosen model: all per-channel
print(f"Weight tensors in {best}: {expected_weights}")
machine = machine_fingerprint()

for held_out in args.held_out:
    allowed = [c for c in CORRUPTIONS if c != held_out]
    precision = f"{best}_mixed_without_{held_out}"
    final = Path("models") / f"{args.model}_{precision}.onnx"
    if final.exists():
        print(f"\n{precision}: already built, skipping")
        continue
    plan = damaged_calibration_plan(len(positions), allowed, seed=PLAN_SEED)
    pictures = np.stack([normalize(crop if name is None else corrupt(crop, name, severity, seed=int(pos)))
                         for crop, pos, (name, severity) in zip(crops, positions, plan, strict=True)])
    calibration = [pictures[i:i + BATCH] for i in range(0, len(pictures), BATCH)]
    print(f"\n{precision}: damaged {sum(n is not None for n, _ in plan)} of {len(plan)} images "
          f"({dict(sorted(Counter(n for n, _ in plan if n).items()))})", flush=True)

    temporary = final.with_name(final.stem + "_building.onnx")
    try:
        to_int8(fp32_path, temporary, calibration, method=method, group_batches=GROUP_BATCHES)
        z = scores(temporary)
        agreement = float(np.mean(z.argmax(1) == fp32_top1))
        weights = weight_quantization(temporary)
        checks = {"loads and gives (256, 1000) scores": z.shape == (len(check), 1000),
                  "all scores finite": bool(np.isfinite(z).all()),
                  f"top-1 agreement with FP32 >= {FAIL_AGREEMENT:.0%} (got {agreement:.1%})":
                      agreement >= FAIL_AGREEMENT,
                  f"weights all per-channel, as in {best}": weights == expected_weights}
        for name, ok in checks.items():
            print(f"    {'ok  ' if ok else 'FAIL'} {name}")
        if not all(checks.values()):
            raise RuntimeError(f"{precision} failed its checks")
        temporary.rename(final)
    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary.with_name(temporary.stem + "_prep.onnx").unlink(missing_ok=True)
        print(f"FAIL: stopped at {precision}; no partial model was kept")
        raise

    final.with_suffix(".json").write_text(json.dumps({
        "model": args.model, "precision": precision, "derived_from": str(fp32_path),
        "licence": MODELS[args.model]["licence"],
        "settings": {**INT8_SETTINGS, "calibration_method": method,
                     "calibration_method_detail": INT8_METHODS[method]["description"],
                     "calibration": {"dataset": DATASET, "dataset_licence": DATASETS[DATASET]["licence"],
                                     "split": "int8_calibration", "n_images": len(positions),
                                     "group_images": GROUP_BATCHES * BATCH,
                                     "held_out_corruption": held_out, "allowed_corruptions": allowed,
                                     "plan_seed": PLAN_SEED,
                                     "corruption_seed": "dataset position of each image",
                                     "plan": [[name, severity] for name, severity in plan],
                                     "excludes": "all other splits, including test"}},
        "file": {"path": str(final), **file_info(final)},
        "sanity_check": {"split": "tuning", "n_images": len(check), "top1_agreement_with_fp32": agreement,
                         "weight_quantization": weights},
        "machine": machine}, indent=2))
    print(f"    saved {final}", flush=True)

print("\nPASS: every model built and passed its checks")
