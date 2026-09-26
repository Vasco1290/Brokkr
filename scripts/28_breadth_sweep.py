"""Task 4.1, the breadth sweep: every model, FP32 and Percentile INT8, on the 13 planned conditions.

Usage:  python scripts/28_breadth_sweep.py                      (the real run: test + conformal_calibration)
        python scripts/28_breadth_sweep.py --split tuning --limit 64 --out <folder>   (a dry run)
Needs:  models/<model>_fp32.onnx and _int8_percentile99.99.onnx with passing build records
        (scripts/24), data/imagenet-1k/, the imagenet-c extra
Writes: <out>/<model>_<precision>_imagenet-1k-val_<split>_<suite>_<corruption>_s<severity>.json
        (schema-2 accuracy record) + .npz (logits, labels, positions), and <out>/run_log.txt

Rules (docs/hypotheses_stage4.md, committed before this runs):
- A model's INT8 is run only if its build record is "usable" (brokkr.schema.check_build_record);
  MobileNetV3-Small's INT8 failed, so it runs in FP32 only.
- Models sharing preprocessing share one keyed cache of clean pictures, and each damaged batch is made
  once for all of them (brokkr.sweep). Damage seed = the image's dataset position.
- Before the first condition: a sample of cached pictures must equal freshly made ones, and (full
  test run only) MobileNetV3-Large's clean and darkness (Brokkr) s5 scores must equal Stage 3's saved
  scores on the first 64 test images. A mismatch stops the run.
- FP32 sanity check, right after each model's clean test run: top-1 within 1.0 point of torchvision's
  published top-1. A model that fails is left out of every later condition (logged).
- The test split: 13 conditions. The conformal_calibration split: clean only (for conformal thresholds).
- Resumable: a (model, precision, condition) whose record and arrays exist with a matching checksum is
  skipped. Test-split reruns only for technical failure, logged with the reason.
- Keeps Windows awake while running (SetThreadExecutionState).
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr.accuracy import accuracy_from_logits, open_image, resize_and_crop
from brokkr.benchmark import make_session
from brokkr.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr.export import MODELS, preprocessing
from brokkr.fingerprint import machine_fingerprint
from brokkr.results import sha256_of
from brokkr.schema import (
    check_build_record,
    condition,
    condition_label,
    make_measurement,
    metric,
    save_measurement,
)
from brokkr.sweep import build_caches, cache_key, damaged_batch, load_cache, normalised

DATASET, BATCH, THREADS, TOLERANCE = "imagenet-1k-val", 32, 4, 0.010
CONDITIONS = ([condition()]
              + [condition("fog", "brokkr", 3), condition("darkness", "brokkr", 5),
                 condition("defocus_blur", "brokkr", 3), condition("noise", "brokkr", 3)]
              + [condition(c, "imagenet-c", s) for c in ("fog", "contrast", "defocus_blur", "gaussian_noise")
                 for s in (3, 5)])
STAGE3_CHECK = {"model": "mobilenet_v3_large", "images": 64,
                "conditions": [condition(), condition("darkness", "brokkr", 5)],
                "precisions": ["fp32", "int8_percentile99.99"]}

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["test", "tuning"], default="test",
                    help="test = the real run (plus clean conformal_calibration); tuning = dry runs only")
parser.add_argument("--models", nargs="+", default=list(MODELS))
parser.add_argument("--limit", type=int, default=None, help="first N images only (dry runs)")
parser.add_argument("--out", default="results/breadth")
args = parser.parse_args()
out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
log_path = out_dir / "run_log.txt"


def log(message: str):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake(on: bool):
    if sys.platform == "win32":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def git(*command) -> str:
    return subprocess.run(["git", *command], capture_output=True, text=True, check=True).stdout.strip()


def build_record(model: str, precision: str) -> dict:
    return json.loads((Path("models") / f"{model}_{precision}.json").read_text(encoding="utf-8"))


def output_path(model, precision, split, cond) -> Path:
    suite = cond["suite"] or "none"
    name = f"{model}_{precision}_{DATASET}_{split}_{suite}_{cond['corruption']}_s{cond['severity']}.json"
    return out_dir / name


def complete(path: Path) -> bool:
    if not (path.exists() and path.with_suffix(".npz").exists()):
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    return record["arrays"]["sha256"] == sha256_of(path.with_suffix(".npz"))


def sanity_check(model: str):
    """FP32 clean test-split top-1 within TOLERANCE of torchvision's published top-1, judged once per
    model from its saved clean record. A model that fails goes into `excluded`."""
    if model in judged:
        return
    judged.add(model)
    record = json.loads(output_path(model, "fp32", "test", condition()).read_text(encoding="utf-8"))
    top1 = record["metrics"]["top1"]["value"]
    published = MODELS[model]["weights"].meta["_metrics"]["ImageNet-1K"]["acc@1"] / 100
    if args.limit is not None:
        log(f"sanity check not judged ({model}): dry run on {record['data']['n_images']} images")
    elif abs(top1 - published) <= TOLERANCE:
        log(f"sanity check PASS ({model}): FP32 clean top-1 {top1:.4f}, torchvision {published:.4f}, "
            f"tolerance {TOLERANCE}")
    else:
        excluded[model] = f"FP32 top-1 {top1:.4f} vs published {published:.4f}"
        log(f"sanity check FAIL ({model}): {excluded[model]}; left out of every later condition")


judged = set()

# 1. Which models and precisions run. A build is used only if its own record says "usable".
plan = {}
for model in args.models:
    precisions = []
    for precision in ("fp32", "int8_percentile99.99"):
        path = Path("models") / f"{model}_{precision}.json"
        status, problems = check_build_record(build_record(model, precision), set()) if path.exists() else \
            ("missing", [])
        if status == "usable" and not problems:
            precisions.append(precision)
        else:
            why = "; ".join([status, *problems])
            print(f"not run: {model} {precision} (build {why})")
    if precisions:
        plan[model] = precisions
groups = defaultdict(list)
for model in plan:
    p = preprocessing(model)
    groups[(p["resize"], p["crop"], p["interpolation"])].append(model)

keep_awake(True)
log(f"breadth sweep at {git('rev-parse', '--short', 'HEAD')} (uncommitted changes: "
    f"{'yes' if git('status', '--porcelain') else 'no'}); split {args.split}; limit {args.limit}; "
    f"{sum(len(v) for v in plan.values())} model-precisions in {len(groups)} preprocessing groups")

files = parquet_files(DATASET)
splits = make_splits(count_images(files))
runtime = {"name": "onnxruntime", "version": ort.__version__, "execution_provider": "CPUExecutionProvider",
           "threads": THREADS}
sessions = {(m, p): make_session(Path("models") / f"{m}_{p}.onnx", THREADS)
            for m, ps in plan.items() for p in ps}
excluded = {}  # model -> reason (FP32 sanity check failed)

jobs = [(args.split, CONDITIONS)]
if args.split == "test":
    jobs.append(("conformal_calibration", [condition()]))
for split, conditions in jobs:
    positions = splits[split][:args.limit] if args.limit else splits[split]
    preps = [{"resize": r, "crop": c, "interpolation": i} for r, c, i in groups]
    cache_paths = build_caches(DATASET, split, positions, files, preps, "data/cache")
    for (group, models), prep, paths in zip(groups.items(), preps, cache_paths, strict=True):
        crops, labels = load_cache(paths, cache_key(DATASET, split, positions, files, prep))

        # 2. Self-check: cached pictures equal freshly made ones (20 evenly spaced images).
        sample = np.linspace(0, len(positions) - 1, 20).astype(int)
        fresh = {int(positions[i]): None for i in sample}
        for n, (image_bytes, _) in enumerate(read_parquet_images(files, positions[sample])):
            fresh[int(positions[sample[n]])] = resize_and_crop(open_image(image_bytes), *group)
        if not all(np.array_equal(crops[i], fresh[int(positions[i])]) for i in sample):
            log(f"STOP: cached pictures differ from fresh ones ({split}, group {group})")
            sys.exit(1)

        # 3. Self-check (full test run only): Stage 3's scores are reproduced exactly.
        if split == "test" and args.limit is None and STAGE3_CHECK["model"] in models:
            n = STAGE3_CHECK["images"]
            for cond in STAGE3_CHECK["conditions"]:
                batch = normalised(damaged_batch(crops, positions, list(range(n)), cond["suite"],
                                                 cond["corruption"], cond["severity"]))
                for precision in STAGE3_CHECK["precisions"]:
                    name = (f"{STAGE3_CHECK['model']}_{precision}_{DATASET}_test_"
                            f"{cond['corruption']}_s{cond['severity']}.npz")
                    stage3 = np.load(Path("results/sweep") / name)["logits"][:n]
                    ours = np.concatenate([sessions[(STAGE3_CHECK["model"], precision)].run(
                        None, {"images": batch[i:i + BATCH]})[0] for i in range(0, n, BATCH)])
                    if not np.array_equal(ours, stage3):
                        log(f"STOP: {precision} {condition_label(cond)} differs from Stage 3 "
                            f"(largest difference {np.abs(ours - stage3).max():.3g})")
                        sys.exit(1)
            log(f"check passed: MobileNetV3-Large reproduces Stage 3's scores exactly on {n} test images "
                "(clean and darkness (Brokkr) s5, FP32 and Percentile INT8)")

        # 4. Every condition: damage each batch once, run every model and precision of the group.
        for cond in conditions:
            label = condition_label(cond)
            if split == "test" and cond["corruption"] != "clean":
                for model in models:
                    sanity_check(model)  # also after a restart, from the saved clean record
            todo = [(m, p) for m in models if m not in excluded for p in plan[m]
                    if not complete(output_path(m, p, split, cond))]
            if not todo:
                log(f"skip (complete): {split} {label}, group {group}")
                continue
            t0 = time.time()
            logits = {job: [] for job in todo}
            for start in range(0, len(positions), BATCH):
                idx = list(range(start, min(start + BATCH, len(positions))))
                batch = normalised(damaged_batch(crops, positions, idx, cond["suite"], cond["corruption"],
                                                 cond["severity"]))
                for job in todo:
                    logits[job].append(sessions[job].run(None, {"images": batch})[0])
            machine = machine_fingerprint()
            for (model, precision), parts in logits.items():
                scores = np.concatenate(parts).astype(np.float32)
                path = output_path(model, precision, split, cond)
                np.savez_compressed(path.with_suffix(".npz"), logits=scores, labels=labels,
                                    positions=np.asarray(positions))
                acc = accuracy_from_logits(scores, labels)["metrics"]
                spec = MODELS[model]
                record = make_measurement(
                    "accuracy", {"name": model, "weights": str(spec["weights"]), "licence": spec["licence"]},
                    precision, runtime, "laptop", machine,
                    {"dataset": DATASET, "split": split, "n_images": len(positions),
                     "licence": DATASETS[DATASET]["licence"]},
                    cond,
                    {"top1": metric(acc["top1"], acc["top1_ci95"]),
                     "top5": metric(acc["top5"], acc["top5_ci95"]),
                     "top1_tied_images": metric(acc["top1_tied_images"])},
                    {"script": "scripts/28_breadth_sweep.py", "batch_size": BATCH, "preprocessing": prep,
                     "damage_seed": "dataset position of each image", "bootstrap_resamples": 1000, "seed": 0,
                     "top1_range_over_tie_breaks": acc["top1_range_over_tie_breaks"],
                     "cache": paths["crops"].name},
                    {"file": path.with_suffix(".npz").name, "sha256": sha256_of(path.with_suffix(".npz"))},
                    [{"file": f"models/{model}_{precision}.onnx",
                      "sha256": build_record(model, precision)["file"]["sha256"]}],
                )
                save_measurement(record, path)
            log(f"done: {split} {label}, group {group}, {len(todo)} model-precisions, "
                f"{(time.time() - t0) / 60:.1f} min")
            if split == "test" and cond["corruption"] == "clean":
                for model in models:
                    sanity_check(model)

keep_awake(False)
log(f"ALL DONE; excluded after the sanity check: {excluded or 'none'}")
