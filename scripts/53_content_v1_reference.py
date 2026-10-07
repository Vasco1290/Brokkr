"""Equality references for the step 3 cross-check: MobileNetV3-Large on the six random conditions with
content-v1 seeds (docs/user_models.md, note of 7 October 2026, evening). Never a result, never on a label.

Usage:  python scripts/53_content_v1_reference.py
        python scripts/53_content_v1_reference.py --dry-run --out <folder outside results/>
        (a tool check: the first 64 tuning images; never a reference)
Needs:  models/mobilenet_v3_large_{fp32,int8_percentile99.99}.onnx (+ build records), data/imagenet-1k/, the
        imagenet-c extra, a clean commit, the laptop on mains power
Writes: results/crosscheck_reference_content_v1/, one file per build and condition, named
        <model>_<precision>_imagenet-1k-val_test_<suite>_<damage>_s<sev>_seed-content-v1_crosscheck-reference.json
        (schema 2, kind "diagnostic") with an .npz of every image's scores; finished records are kept, so the
        same command can be run again after a stop

What it does: the study's code for the 4.1 records (the same clean-picture caches, preprocessing, 8 threads,
thread spinning off, batches of 32) on the 10,000 test images, for the six conditions that draw random
numbers,
with only the seed source swapped: content-v1 (from the SHA-256 of each image's original JPEG bytes, as stored
in the dataset; brokkr_edge.seeds) instead of the dataset position. Before anything is measured, a check: with
position seeds, the seeded damage equals the study's own (brokkr_edge.sweep.damaged_batch) on the first batch
of every condition; otherwise the script stops. The test images serve only as an equality reference: these
records set no number, threshold or setting.
"""

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr_edge.accuracy import accuracy_from_logits
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.fingerprint import git_info, machine_fingerprint, power_state
from brokkr_edge.model_list import load_model_list
from brokkr_edge.results import sha256_of, written_atomically
from brokkr_edge.schema import condition, condition_label, make_measurement, metric, save_measurement
from brokkr_edge.seeds import content_seed, damaged_batch_seeded
from brokkr_edge.sweep import (
    MIN_FREE_GB,
    build_caches,
    cache_key,
    damaged_batch,
    duration,
    free_gb,
    load_cache,
    normalised,
    seed_for,
)
from brokkr_edge.test_run import complete, keep_awake, usable_precisions

MODEL, DATASET = "mobilenet_v3_large", "imagenet-1k-val"
PRECISIONS = ("fp32", "int8_percentile99.99")
THREADS, BATCH = 8, 32
RANDOM_CONDITIONS = [condition("fog", "brokkr", 3), condition("noise", "brokkr", 3)] + [
    condition(c, "imagenet-c", s) for c in ("fog", "gaussian_noise") for s in (3, 5)]
SUFFIX = "_seed-content-v1_crosscheck-reference"
PURPOSE = ("equality reference for the step 3 cross-check only (docs/user_models.md, note of 7 October 2026, "
           "evening): never a result, never on a label; the test images set no number here")
DRY_RUN_IMAGES = 64

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="tool check on the first 64 tuning images")
parser.add_argument("--out", default=None, help="with --dry-run: a folder outside results/")
args = parser.parse_args()
if args.dry_run:
    if not args.out or Path(args.out).resolve().is_relative_to(Path("results").resolve()):
        sys.exit("STOP: --dry-run needs --out <folder outside results/>")
    SPLIT, OUT = "tuning", Path(args.out)
else:
    SPLIT, OUT = "test", Path("results/crosscheck_reference_content_v1")

if git_info()["dirty"] and not args.dry_run:
    sys.exit("STOP: uncommitted changes. The references must come from a clean commit.")
if power_state()["on_ac_power"] is False:
    sys.exit("STOP: the laptop is on battery. Plug it in and run the same command again.")
if usable_precisions(MODEL) != list(PRECISIONS):
    sys.exit(f"STOP: {MODEL} needs usable FP32 and INT8 builds in models/")
OUT.mkdir(parents=True, exist_ok=True)
if free_gb(OUT) < MIN_FREE_GB:
    sys.exit(f"STOP: less than {MIN_FREE_GB} GB free on the disk of {OUT}")

entry = load_model_list()[MODEL]
prep = {k: entry["input"]["preprocessing"][k] for k in ("resize", "crop", "interpolation")}
files = parquet_files(DATASET)
positions = make_splits(count_images(files))[SPLIT]
if args.dry_run:
    positions = positions[:DRY_RUN_IMAGES]
cache = build_caches(DATASET, SPLIT, positions, files, [prep], "data/cache")[0]
crops, labels = load_cache(cache, cache_key(DATASET, SPLIT, positions, files, prep))
originals = list(read_parquet_images(files, positions))  # (original JPEG bytes, label), in split order
image_sha256 = [hashlib.sha256(image_bytes).hexdigest() for image_bytes, _ in originals]
if len(originals) != len(positions) or [label for _, label in originals] != labels.tolist():
    sys.exit("STOP: the dataset's images do not line up with the cached pictures and labels")
del originals
model_files = {p: Path("models") / f"{MODEL}_{p}.onnx" for p in PRECISIONS}
sessions = {p: make_session(model_files[p], THREADS, spinning=False) for p in PRECISIONS}
print(f"content-v1 references: {MODEL} ({', '.join(PRECISIONS)}), {len(positions)} {SPLIT} images, "
      f"{len(RANDOM_CONDITIONS)} random conditions; {THREADS} threads, spinning off, batch {BATCH}; in {OUT}")

first = list(range(min(BATCH, len(positions))))
for cond in RANDOM_CONDITIONS:  # check before measuring: position seeds give exactly the study's damage
    study = damaged_batch(crops, positions, first, cond["suite"], cond["corruption"], cond["severity"])
    seeded = damaged_batch_seeded(crops, first, cond["suite"], cond["corruption"], cond["severity"],
                                  [seed_for(positions[i]) for i in first])
    if not np.array_equal(study, seeded):
        sys.exit(f"STOP: seeded damage differs from the study's for {condition_label(cond)}")
print("check before measuring: PASS (position seeds reproduce the study's damage on the first batch of every "
      "condition)")

runtime = {"name": "onnxruntime", "version": ort.__version__, "execution_provider": "CPUExecutionProvider",
           "threads": THREADS, "spinning": "off"}
keep_awake(True)
try:
    for cond in RANDOM_CONDITIONS:
        stem = f"{DATASET}_{SPLIT}_{cond['suite']}_{cond['corruption']}_s{cond['severity']}"
        name = {p: f"{MODEL}_{p}_{stem}{SUFFIX}" for p in PRECISIONS}
        paths = {p: OUT / f"{name[p]}.json" for p in PRECISIONS}
        todo = [p for p in PRECISIONS if not complete(paths[p])]
        if not todo:
            print(f"skip (complete): {condition_label(cond)}")
            continue
        started = time.time()
        seeds = [content_seed(s, cond["suite"], cond["corruption"]) for s in image_sha256]
        scores = {p: [] for p in todo}
        for start in range(0, len(positions), BATCH):
            idx = list(range(start, min(start + BATCH, len(positions))))
            batch = normalised(damaged_batch_seeded(crops, idx, cond["suite"], cond["corruption"],
                                                    cond["severity"], [seeds[i] for i in idx]))
            for p in todo:
                scores[p].append(sessions[p].run(None, {"images": batch})[0])
        machine = machine_fingerprint()
        for p in todo:
            logits = np.concatenate(scores[p]).astype(np.float32)
            npz = paths[p].with_suffix(".npz")
            with written_atomically(npz) as tmp, tmp.open("wb") as f:
                np.savez_compressed(f, logits=logits, labels=labels, positions=np.asarray(positions),
                                    seeds=np.asarray(seeds, dtype=np.uint32))
            acc = accuracy_from_logits(logits, labels)["metrics"]
            record = make_measurement(
                "diagnostic", {"name": MODEL, "weights": entry["weights"], "licence": entry["licence"]}, p,
                runtime, "laptop", machine,
                {"dataset": DATASET, "split": SPLIT, "n_images": len(positions),
                 "licence": DATASETS[DATASET]["licence"]},
                cond,
                {"top1": metric(acc["top1"], acc["top1_ci95"]),
                 "top1_tied_images": metric(acc["top1_tied_images"])},
                {"script": "scripts/53_content_v1_reference.py", "purpose": PURPOSE,
                 "seed_scheme": "content-v1",
                 "damage_seed": "content-v1: first 4 bytes of SHA-256('<image SHA-256>:<suite>/<damage>')",
                 "batch_size": BATCH, "preprocessing": prep, "cache": cache["crops"].name,
                 "dry_run": bool(args.dry_run)},
                {"file": npz.name, "sha256": sha256_of(npz)},
                [{"file": model_files[p].as_posix(), "sha256": sha256_of(model_files[p])}])
            save_measurement(record, paths[p])
        print(f"done: {condition_label(cond)}, {len(todo)} builds, {duration(time.time() - started)}")
finally:
    keep_awake(False)
print(f"{len(RANDOM_CONDITIONS) * len(PRECISIONS)} reference records in {OUT} "
      f"({shutil.disk_usage(OUT).free / 1e9:.1f} GB free)")
