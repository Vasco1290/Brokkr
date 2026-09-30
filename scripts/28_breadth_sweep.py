"""Task 4.1, the breadth sweep: every model, FP32 and Percentile INT8, on the 13 planned conditions.

Usage:  python scripts/28_breadth_sweep.py                      (the real run: test + conformal_calibration)
        python scripts/28_breadth_sweep.py --split tuning --limit 64 --out <folder>   (a dry run)
Needs:  models/<model>_fp32.onnx and _int8_percentile99.99.onnx with passing build records
        (scripts/24), data/imagenet-1k/, the imagenet-c extra
Writes: <out>/<model>_<precision>_imagenet-1k-val_<split>_<suite>_<corruption>_s<severity>.json
        (schema-2 accuracy record) + .npz (logits, labels, positions), and <out>/run_log.txt

Rules (docs/hypotheses_stage4.md, committed before this runs):
- A model's INT8 is run only if its build record is "usable" (brokkr_edge.schema.check_build_record);
  MobileNetV3-Small's INT8 failed, so it runs in FP32 only.
- Order: smallest measured run time first, ResNet-50 and ConvNeXt-Tiny last. Neighbouring models with
  the same preprocessing form one pass and share each damaged batch; every group has one keyed cache
  of clean pictures (brokkr_edge.sweep). Damage seed = the image's dataset position. 8 threads (checked
  bit-identical to 4 threads for every model, scripts/29_thread_check.py).
- Before the first condition: a sample of cached pictures must equal freshly made ones, and (full
  test run only) MobileNetV3-Large's clean and darkness (Brokkr) s5 scores must equal Stage 3's saved
  scores on the first 64 test images. A mismatch stops the run.
- FP32 sanity check, right after each model's clean test run: top-1 within 1.0 point of torchvision's
  published top-1. A model that fails is left out of every later condition (logged).
- The test split: 13 conditions. The conformal_calibration split: clean only (for conformal thresholds).
- Resumable: a (model, precision, condition) whose record and arrays exist with a matching checksum is
  skipped. Test-split reruns only for technical failure, logged with the reason.
- Keeps Windows awake while running (SetThreadExecutionState).
- Safe to interrupt: every file is written under a .tmp name and renamed only when complete; a step
  counts as done only if its record and score file exist and the checksum matches.
- Disk space: before building caches (counting the space they will take) and before every step, the
  free space must stay at or above --min-free-gb (default 8). If not, the run stops cleanly (exit code
  3); free some space and rerun the same command.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr_edge.accuracy import accuracy_from_logits, open_image, resize_and_crop
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.export import MODELS, preprocessing
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.results import sha256_of, written_atomically
from brokkr_edge.schema import (
    check_build_record,
    condition,
    condition_label,
    make_measurement,
    metric,
    save_measurement,
)
from brokkr_edge.sweep import (
    MIN_FREE_GB,
    build_caches,
    cache_key,
    damaged_batch,
    duration,
    estimated_step_seconds,
    free_gb,
    load_cache,
    missing_cache_bytes,
    normalised,
    slow_warning,
)

DATASET, BATCH, TOLERANCE = "imagenet-1k-val", 32, 0.010
# 8 threads: on 26 September 2026 every model's FP32 and Percentile INT8 scores were bit-identical with
# 4 and 8 threads on 64 tuning images, and 8 was faster for all (scripts/29_thread_check.py).
THREADS = 8
# Smallest measured 4.1 run time first (results/profile/*_all10.json, 26 September 2026).
MODEL_ORDER = ["mobilenet_v3_small", "shufflenet_v2_x1_0", "mnasnet1_0", "mobilenet_v2", "mobilenet_v3_large",
               "regnet_y_400mf", "efficientnet_b0", "resnet18", "resnet50", "convnext_tiny"]
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
# Spinning off (threads sleep between runs): on 26 September 2026 every score was bit-identical with
# spinning on and off, and off was 2.46x faster in the sweep's pattern (scripts/30_spinning_check.py).
parser.add_argument("--spinning", choices=["on", "off"], default="off",
                    help="ONNX Runtime threads busy-wait between runs (on) or sleep (off, the default)")
parser.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB,
                    help="stop cleanly if free disk space would fall below this")
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
    """Done = the record and its score file exist under their real names and the checksum matches."""
    if not (path.exists() and path.with_suffix(".npz").exists()):
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record["arrays"]["sha256"] == sha256_of(path.with_suffix(".npz"))
    except (ValueError, KeyError, TypeError):
        return False


def load_estimates():
    """Images per second from the 26 September profiles (results/profile), for step-time estimates.

    Model rates were measured with 4 threads, one session at a time; for 8 threads each is scaled by
    that model's measured 8/4-thread ratio (scripts/29). Damage and normalising rates come from the
    MobileNetV3-Large profile. Returns None if a profile file is missing (then no estimates are logged).
    """
    folder = Path("results/profile")
    try:
        rates = {}
        profiles = list(folder.glob("*_pipeline_profile_all10.json"))
        for f in profiles + [folder / "convnext_tiny_pipeline_profile_int8.json"]:
            r = json.loads(f.read_text(encoding="utf-8"))
            for precision, v in r["metrics"]["inference"].items():
                rates[(r["model"], precision)] = v["images_per_second_median_pass"]
        if THREADS != 4:
            for f in folder.glob("*_thread_check.json"):
                r = json.loads(f.read_text(encoding="utf-8"))
                for precision, v in r["metrics"].items():
                    m = v["images_per_second_median"]
                    if (r["model"], precision) in rates and str(THREADS) in m:
                        rates[(r["model"], precision)] *= m[str(THREADS)] / m["4"]
        steps = json.loads((folder / "mobilenet_v3_large_pipeline_profile_all10.json").read_text(
            encoding="utf-8"))["metrics"]["steps"]
    except (OSError, KeyError, ValueError):
        return None
    damage = {k: v["images_per_second"] for k, v in steps.items() if "(" in k}
    return rates, damage, steps["normalise"]["images_per_second"]


def step_estimate(jobs, cond, n_images):
    if estimates is None:
        return None
    rates, damage, normalise = estimates
    damage_rate = None
    if cond["corruption"] != "clean":
        suite = {"brokkr": "Brokkr", "imagenet-c": "ImageNet-C"}[cond["suite"]]
        damage_rate = damage.get(f"{cond['corruption']} ({suite}) s{cond['severity']}")
        if damage_rate is None:
            return None
    return estimated_step_seconds(jobs, n_images, rates, damage_rate, normalise)


def check_disk(step: str, needed_bytes: int = 0):
    """Stop cleanly if free space (after `needed_bytes` more) would fall below --min-free-gb."""
    free = free_gb(out_dir)
    if free - needed_bytes / 1e9 < args.min_free_gb:
        log(f"STOP (disk): {free:.1f} GB free before {step}"
            + (f", which needs {needed_bytes / 1e9:.1f} GB" if needed_bytes else "")
            + f"; the minimum is {args.min_free_gb:.1f} GB. Free some space and rerun the same command: "
            "finished steps are kept.")
        keep_awake(False)
        sys.exit(3)


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
# 2. Order: smallest measured 4.1 run time first, ResNet-50 and ConvNeXt-Tiny last. Neighbouring models
# with the same preprocessing form one pass, so each damaged batch is made once per pass.
passes = []
for model in [m for m in MODEL_ORDER if m in plan] + [m for m in plan if m not in MODEL_ORDER]:
    p = preprocessing(model)
    key = (p["resize"], p["crop"], p["interpolation"])
    if passes and passes[-1][0] == key:
        passes[-1][1].append(model)
    else:
        passes.append((key, [model]))
group_keys = list(dict.fromkeys(key for key, _ in passes))

keep_awake(True)
log(f"breadth sweep at {git('rev-parse', '--short', 'HEAD')} (uncommitted changes: "
    f"{'yes' if git('status', '--porcelain') else 'no'}); split {args.split}; limit {args.limit}; "
    f"{THREADS} threads, spinning {args.spinning}; {sum(len(v) for v in plan.values())} model-precisions "
    f"in {len(passes)} passes: "
    + " | ".join(", ".join(models) for _, models in passes))

files = parquet_files(DATASET)
splits = make_splits(count_images(files))
runtime = {"name": "onnxruntime", "version": ort.__version__, "execution_provider": "CPUExecutionProvider",
           "threads": THREADS, "spinning": args.spinning}
sessions = {(m, p): make_session(Path("models") / f"{m}_{p}.onnx", THREADS, spinning=args.spinning == "on")
            for m, ps in plan.items() for p in ps}
estimates = load_estimates()
excluded = {}  # model -> reason (FP32 sanity check failed)

jobs = [(args.split, CONDITIONS)]
if args.split == "test":
    jobs.append(("conformal_calibration", [condition()]))
for split, conditions in jobs:
    positions = splits[split][:args.limit] if args.limit else splits[split]
    preps = {key: {"resize": key[0], "crop": key[1], "interpolation": key[2]} for key in group_keys}
    check_disk(f"building the {split} caches",
               missing_cache_bytes(DATASET, split, positions, files, list(preps.values()), "data/cache"))
    built = build_caches(DATASET, split, positions, files, list(preps.values()), "data/cache")
    cache = {key: paths for key, paths in zip(group_keys, built, strict=True)}
    loaded = {key: load_cache(cache[key], cache_key(DATASET, split, positions, files, preps[key]))
              for key in group_keys}

    # Pre-flight 1: cached pictures equal freshly made ones (20 evenly spaced images, every group;
    # the 20 photos are read from the dataset once).
    sample = np.linspace(0, len(positions) - 1, 20).astype(int)
    photos = [open_image(b) for b, _ in read_parquet_images(files, positions[sample])]
    for key in group_keys:
        crops = loaded[key][0]
        if not all(np.array_equal(crops[i], resize_and_crop(photo, *key)) for i, photo in zip(sample, photos,
                                                                                            strict=True)):
            log(f"STOP: cached pictures differ from fresh ones ({split}, group {key})")
            sys.exit(1)
    log(f"check passed: cached pictures equal fresh ones ({split}, 20 images, {len(group_keys)} groups)")

    # Pre-flight 2 (full test run only): Stage 3's scores are reproduced exactly, before any condition.
    if split == "test" and args.limit is None and STAGE3_CHECK["model"] in plan:
        p = preprocessing(STAGE3_CHECK["model"])
        crops = loaded[(p["resize"], p["crop"], p["interpolation"])][0]
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
            f"(clean and darkness (Brokkr) s5, FP32 and Percentile INT8, {THREADS} threads)")

    for group, models in passes:
        crops, labels = loaded[group]
        prep, paths = preps[group], cache[group]

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
            check_disk(f"{split} {label}, group {group}")
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
                with written_atomically(path.with_suffix(".npz")) as tmp, tmp.open("wb") as f:
                    np.savez_compressed(f, logits=scores, labels=labels, positions=np.asarray(positions))
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
            took = time.time() - t0
            estimate = step_estimate(todo, cond, len(positions))
            vs = f"estimate {duration(estimate)}, {took / estimate:.1f}x" if estimate else "no estimate"
            log(f"done: {split} {label}, group {group}, {len(todo)} model-precisions, "
                f"{duration(took)} ({vs})")
            warning = slow_warning(took, estimate)
            if warning:
                log(warning)
            if split == "test" and cond["corruption"] == "clean":
                for model in models:
                    sanity_check(model)

keep_awake(False)
log(f"ALL DONE; excluded after the sanity check: {excluded or 'none'}")
