"""Build best INT8 with its final layer's output left unrounded (task 3.4).

Usage:  python scripts/15_int8_unrounded_output.py [--model NAME]
Needs:  models/<model>_fp32.onnx, results/choices/<model>_int8_method.json (scripts/11), the chosen
        model's clean tuning result (scripts/03), the tuning image cache (scripts/08 --split tuning)
        and data/imagenet-1k/
Writes: models/<model>_<best>_unrounded.onnx (+ .json record)

Everything is as for "best INT8" (same method, the same 512 int8_calibration images, groups of 128,
per-channel int8 weights) except one setting: the final layer (Gemm, the classifier) keeps its
OUTPUT in float instead of rounding it to 8 bits (onnxruntime's OpTypesToExcludeOutputQuantization).
Its weights stay int8. Rounded outputs can make two classes tie exactly; unrounded ones should not.

Checks before the model is kept (the run stops otherwise, leaving no partial file): it loads, gives
finite scores of the right shape, agrees with FP32 on at least 20% of 256 clean tuning images, has
the same per-channel weights as best INT8, and its final layer's output is really not rounded.
Tool check afterwards (not a result): tied top scores on the clean tuning images, with and without
rounding. H11 itself (ties and E-AURC on the TEST split) is judged in the final run, task 3.7.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx

from brokkr_edge.accuracy import accuracy_from_logits, normalize, open_image, preprocess
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.export import MODELS, file_info
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.quantize import CALIBRATION_BATCH as BATCH
from brokkr_edge.quantize import CALIBRATION_GROUP_BATCHES as GROUP_BATCHES
from brokkr_edge.quantize import INT8_METHODS, INT8_SETTINGS, check_int8_build, to_int8, weight_quantization
from brokkr_edge.results import load_arrays

DATASET = "imagenet-1k-val"
UNROUNDED_OPS = ["Gemm"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
args = parser.parse_args()

fp32_path = Path("models") / f"{args.model}_fp32.onnx"
choice_path = Path("results/choices") / f"{args.model}_int8_method.json"
crops_path = Path("data/cache") / f"{DATASET}_tuning_crops.npy"
for path in (fp32_path, choice_path, crops_path):
    if not path.exists():
        sys.exit(f"FAIL: {path} not found (see Needs in this script's docstring)")
choice = json.loads(choice_path.read_text())["metrics"]
best, method = choice["chosen"], choice["chosen_method"]
best_path = Path("models") / f"{args.model}_{best}.onnx"
precision = f"{best}_unrounded"
final = Path("models") / f"{args.model}_{precision}.onnx"
if final.exists():
    sys.exit(f"{final} already exists; delete it first to rebuild")


def final_output_rounded(path) -> bool:
    """True if the output of the model's last Gemm goes into a QuantizeLinear (is rounded to 8 bits)."""
    graph = onnx.load(str(path)).graph
    gemm_output = [n for n in graph.node if n.op_type == "Gemm"][-1].output[0]
    return any(n.op_type == "QuantizeLinear" and n.input[0] == gemm_output for n in graph.node)


files = parquet_files(DATASET)
splits = make_splits(count_images(files))
print(f"Loading {len(splits['int8_calibration'])} calibration images...")
images = np.stack([preprocess(open_image(image))
                   for image, _ in read_parquet_images(files, splits["int8_calibration"])])
crops = np.load(crops_path, mmap_mode="r")
check = np.stack([normalize(c) for c in crops[:256]])
fp32_top1 = make_session(fp32_path, 4).run(None, {"images": check})[0].argmax(1)
expected_weights = weight_quantization(best_path)

print(f"Building {precision}: {best} with the {UNROUNDED_OPS} output left unrounded...", flush=True)
temporary = final.with_name(final.stem + "_building.onnx")
try:
    to_int8(fp32_path, temporary, [images[i:i + BATCH] for i in range(0, len(images), BATCH)],
            method=method, group_batches=GROUP_BATCHES, unrounded_output_ops=UNROUNDED_OPS)
    checks = check_int8_build(temporary, check, fp32_top1, expected_weights)
    checks["final layer output not rounded (and it is rounded in best INT8)"] = (
        not final_output_rounded(temporary) and final_output_rounded(best_path))
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

# Tool check on clean tuning images: tied top scores with and without rounding.
session = make_session(final, num_threads=4)
batches = (np.stack([normalize(c) for c in crops[i:i + BATCH]]) for i in range(0, len(crops), BATCH))
logits = np.concatenate([session.run(None, {"images": b})[0] for b in batches]).astype(np.float32)
rounded = load_arrays(Path("results/accuracy") / f"{args.model}_{best}_{DATASET}_tuning.json")
ties = {name: accuracy_from_logits(z, rounded["labels"])["metrics"]["top1_tied_images"]
        for name, z in (("rounded", rounded["logits"]), ("unrounded", logits))}
sizes = {name: file_info(path)["size_bytes"] for name, path in (("rounded", best_path), ("unrounded", final))}
print(f"\nClean tuning images ({len(logits):,}): tied top scores {ties['rounded']} with rounding, "
      f"{ties['unrounded']} without")
print(f"File size: {sizes['rounded'] / 1e6:.3f} MB -> {sizes['unrounded'] / 1e6:.3f} MB "
      f"({(sizes['unrounded'] / sizes['rounded'] - 1) * 100:+.2f}%)")

final.with_suffix(".json").write_text(json.dumps({
    "model": args.model, "precision": precision, "derived_from": str(fp32_path),
    "licence": MODELS[args.model]["licence"],
    "settings": {**INT8_SETTINGS, "calibration_method": method,
                 "calibration_method_detail": INT8_METHODS[method]["description"],
                 "unrounded_output_ops": UNROUNDED_OPS,
                 "calibration": {"dataset": DATASET, "dataset_licence": DATASETS[DATASET]["licence"],
                                 "split": "int8_calibration", "n_images": len(images),
                                 "group_images": GROUP_BATCHES * BATCH,
                                 "excludes": "all other splits, including test"}},
    "file": {"path": str(final), **file_info(final)},
    "sanity_check": {"split": "tuning", "checks": checks,
                     "tied_top_scores_clean_tuning": ties, "file_size_bytes": sizes},
    "machine": machine_fingerprint()}, indent=2))
print(f"Saved {final}")
passed = ties["unrounded"] == 0
print("PASS" if passed else "FAIL: the unrounded model still has tied top scores")
sys.exit(0 if passed else 1)
