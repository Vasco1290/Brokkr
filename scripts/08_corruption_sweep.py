"""Run every model on every corrupted version of one split's images and save all class scores.

Usage:  python scripts/08_corruption_sweep.py [--split test|tuning|conformal_calibration]
            [--precision fp32 fp16 int8] [--corruptions fog defocus_blur ...]
            [--severities 0 1 2 3 4 5] [--limit N] [--out DIR]
        test:                  the 10,000 images every reported result is measured on (default)
        tuning:                5,000 images for choosing Stage 3 settings (never the test split)
        conformal_calibration: 5,000 images for tuning conformal thresholds (Stage 3: also damaged)
Needs:  models/<model>_<precision>.onnx and data/imagenet-1k/
Writes: <out>/<model>_<precision>_imagenet-1k-val_<split>_<corruption>_s<severity>.json (+ .npz logits)
        (default out: results/sweep). Severity 0 = clean images.

How it runs:
- The split's images are decoded and resized once and cached in data/cache/ (about 1.5 GB for the
  10,000 test images, 0.75 GB for a 5,000-image split).
- For each condition, each image is corrupted once and the same corrupted image goes to every
  precision, so precisions are compared on identical inputs.
- Each image's corruption seed is its position in the dataset: every run gets the same fog, noise
  and blur direction.
- Resumable: conditions whose result file already exists are skipped, so an interrupted overnight
  run can just be started again.

Check at the end: the clean condition (severity 0) must reproduce the saved logits from
scripts/03_evaluate_accuracy.py for the same split and images, so the cached path matches the
validated one. (Run 03 with the same --split first, or the check is skipped and the run FAILS.)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from brokkr_edge.accuracy import accuracy_from_logits, normalize, open_image, resize_and_crop
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.results import load_arrays, make_record, save_arrays, save_record
from brokkr_edge.shift import CORRUPTIONS, corrupt

DATASET = "imagenet-1k-val"
BATCH = 32

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="mobilenet_v3_large")
parser.add_argument("--split", default="test", choices=["test", "tuning", "conformal_calibration"])
parser.add_argument("--precision", nargs="+", default=["fp32", "fp16", "int8"])
parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTIONS), choices=list(CORRUPTIONS))
parser.add_argument("--severities", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
parser.add_argument("--limit", type=int, default=None, help="only the first N images (for trials)")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--out", default="results/sweep")
args = parser.parse_args()

paths = {p: Path("models") / f"{args.model}_{p}.onnx" for p in args.precision}
for path in paths.values():
    if not path.exists():
        sys.exit(f"FAIL: {path} not found. Run scripts/01_export_model.py and 04_quantize.py first.")

# 1. Decode and resize the split's images once (cached on disk, reused by every condition).
files = parquet_files(DATASET)
positions = make_splits(count_images(files))[args.split]
cache_dir = Path("data/cache")
crops_path = cache_dir / f"{DATASET}_{args.split}_crops.npy"
labels_path = cache_dir / f"{DATASET}_{args.split}_labels.npy"
if not crops_path.exists():
    print(f"Caching {len(positions):,} resized {args.split} images to {crops_path} (one-off)...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    crops = np.lib.format.open_memmap(crops_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8,
                                      shape=(len(positions), 224, 224, 3))
    labels = np.empty(len(positions), dtype=np.int64)
    for i, (image, label) in enumerate(read_parquet_images(files, positions)):
        crops[i], labels[i] = resize_and_crop(open_image(image)), label
    crops.flush()
    del crops
    crops_path.with_suffix(".tmp.npy").rename(crops_path)  # only a complete cache gets the real name
    np.save(labels_path, labels)
crops = np.load(crops_path, mmap_mode="r")  # read from disk as needed, not all into memory
labels = np.load(labels_path)
n = args.limit or len(positions)

sessions = {p: make_session(path, args.threads) for p, path in paths.items()}
machine = machine_fingerprint()
out_dir = Path(args.out)
conditions = [("clean", 0)] if 0 in args.severities else []
conditions += [(c, s) for c in args.corruptions for s in args.severities if s > 0]

# 2. Every condition: corrupt each image once, run every precision on the same batch.
start = time.time()
for done, (name, severity) in enumerate(conditions):
    outputs = {p: out_dir / f"{args.model}_{p}_{DATASET}_{args.split}_{name}_s{severity}.json" for p in paths}
    todo = [p for p, o in outputs.items() if not o.exists()]
    if not todo:
        print(f"[{done + 1}/{len(conditions)}] {name} s{severity}: already done, skipping")
        continue
    t0 = time.time()
    logits = {p: [] for p in todo}
    for b in range(0, n, BATCH):
        batch = np.stack([normalize(crops[i] if severity == 0 else
                                    corrupt(np.asarray(crops[i]), name, severity, seed=int(positions[i])))
                          for i in range(b, min(b + BATCH, n))])
        for p in todo:
            logits[p].append(sessions[p].run(None, {"images": batch})[0])

    top1 = {}
    for p in todo:
        z = np.concatenate(logits[p]).astype(np.float32)
        result = accuracy_from_logits(z, labels[:n])
        top1[p] = result["metrics"]["top1"]
        result["settings"] = {"n_images": n, "batch_size": BATCH, "num_threads": args.threads,
                              "bootstrap_resamples": 1000, "seed": 0, "dataset": DATASET,
                              "dataset_licence": DATASETS[DATASET]["licence"], "split": args.split,
                              "corruption": name, "severity": severity,
                              "corruption_seed": "dataset position of each image"}
        record = make_record("accuracy", args.model, p, result, machine)
        save_arrays(record, outputs[p], logits=z, labels=labels[:n], positions=positions[:n])
        save_record(record, outputs[p])
    elapsed = time.time() - start
    remaining = elapsed / (done + 1) * (len(conditions) - done - 1)
    scores = "  ".join(f"{p} {acc:.1%}" for p, acc in top1.items())
    print(f"[{done + 1}/{len(conditions)}] {name} s{severity}: {scores}  "
          f"({time.time() - t0:.0f} s; about {remaining / 60:.0f} min left)", flush=True)

# 3. Check: the clean condition must match the validated results for the same split.
passed = True
for p in paths:
    clean = out_dir / f"{args.model}_{p}_{DATASET}_{args.split}_clean_s0.json"
    reference = Path("results/accuracy") / f"{args.model}_{p}_{DATASET}_{args.split}.json"
    if not reference.exists():
        print(f"clean check {p}: no {reference.name} to compare with "
              f"(run scripts/03_evaluate_accuracy.py --precision {p} --split {args.split} first)")
        passed = False
    elif clean.exists():
        a, r = load_arrays(clean), load_arrays(reference)
        m = len(a["labels"])
        same_images = np.array_equal(a["positions"], r["positions"][:m])
        max_diff = float(np.abs(a["logits"] - r["logits"][:m]).max())
        same_answers = np.array_equal(a["logits"].argmax(1), r["logits"][:m].argmax(1))
        print(f"clean check {p}: same images {same_images}, largest logit difference {max_diff:.2e}, "
              f"same top-1 answers {same_answers}")
        passed &= same_images and same_answers and max_diff < 1e-3
    else:
        print(f"clean check {p}: skipped (severity 0 not run, no {clean.name})")

print(f"\nFinished in {(time.time() - start) / 60:.1f} min")
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
