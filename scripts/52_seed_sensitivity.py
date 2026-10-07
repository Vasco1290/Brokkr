"""How much does top-1 move when only the damage seed changes? (D18 c; docs/user_models.md, notes of 7 October
2026, method fixed before this script existed.) Prints the proposed margin M.

Usage:  python scripts/52_seed_sensitivity.py
        python scripts/52_seed_sensitivity.py --dry-run --out <folder outside results/>   (a tool check: the
        first 64 tuning images, which are not part of the measurement; never a result)
Needs:  models/mobilenet_v3_large_{fp32,int8_percentile99.99}.onnx (+ build records), data/imagenet-1k/, the
        imagenet-c extra, a clean commit, the laptop on mains power
Writes: results/seed_sensitivity/<model>_<precision>_conformal_calibration_<suite>_<damage>_s<severity>.json
        (schema 2, kind "diagnostic": never a result), each with an .npz of the per-image answers;
        results/checks/seed_sensitivity.json (the 12 cells, M and what it means)

Method (fixed in the note of 7 October 2026 before anything was measured):
- MobileNetV3-Large, its FP32 file and its Percentile 99.99 INT8 file; the 5,000 conformal_calibration images
  (never test images); the study's preprocessing, threads (8), spinning (off) and batch size (32);
- the six conditions that draw random numbers, each run twice on the same images: with the study's seeds
  ("position-study": the dataset position) and with "content-v1" seeds (from the SHA-256 of each image's
  original JPEG bytes, as stored in the dataset);
- per build and condition (12 cells): top-1 under content-v1 minus top-1 under position-study, counted in
  whole images, with its paired bootstrap 95% interval (1,000 resamples, seed 0);
- M = the largest absolute end of the 12 intervals, in points, rounded up to the next 0.1 point. If M is above
  1.5 points, the work stops for review instead of adopting M (H, 7 October 2026).
Before measuring, a check: with position seeds, the seeded damage equals the study's own damage
(brokkr_edge.sweep.damaged_batch) on the first batch of every condition; otherwise the script stops.
"""

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr_edge.accuracy import bootstrap_ci
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.fingerprint import git_info, machine_fingerprint, power_state
from brokkr_edge.label import paired
from brokkr_edge.model_list import load_model_list
from brokkr_edge.results import make_record, save_record, sha256_of, written_atomically
from brokkr_edge.schema import (
    check_record,
    condition,
    condition_label,
    make_measurement,
    metric,
    save_measurement,
)
from brokkr_edge.seeds import content_seed, damaged_batch_seeded
from brokkr_edge.sweep import (
    build_caches,
    cache_key,
    damaged_batch,
    duration,
    load_cache,
    normalised,
    seed_for,
)
from brokkr_edge.test_run import usable_precisions

MODEL, DATASET, SPLIT = "mobilenet_v3_large", "imagenet-1k-val", "conformal_calibration"
PRECISIONS = ("fp32", "int8_percentile99.99")
THREADS, BATCH = 8, 32
RANDOM_CONDITIONS = [condition("fog", "brokkr", 3), condition("noise", "brokkr", 3)] + [
    condition(c, "imagenet-c", s) for c in ("fog", "gaussian_noise") for s in (3, 5)]
STOP_ABOVE_POINTS = 1.5
DRY_RUN_IMAGES = 64

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="tool check on the first 64 tuning images")
parser.add_argument("--out", default=None, help="with --dry-run: a folder outside results/")
args = parser.parse_args()
if args.dry_run:
    if not args.out or Path(args.out).resolve().is_relative_to(Path("results").resolve()):
        sys.exit("STOP: --dry-run needs --out <folder outside results/>")
    SPLIT = "tuning"
    OUT, SUMMARY = Path(args.out), Path(args.out) / "seed_sensitivity.json"
else:
    OUT, SUMMARY = Path("results/seed_sensitivity"), Path("results/checks/seed_sensitivity.json")

if git_info()["dirty"] and not args.dry_run:
    sys.exit("STOP: uncommitted changes. The measurement must come from a clean commit.")
if power_state()["on_ac_power"] is False:
    sys.exit("STOP: the laptop is on battery. Plug it in and run the same command again.")
if usable_precisions(MODEL) != list(PRECISIONS):
    sys.exit(f"STOP: {MODEL} needs usable FP32 and INT8 builds in models/")

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
print(f"seed sensitivity: {MODEL} ({', '.join(PRECISIONS)}), {len(positions)} {SPLIT} images, "
      f"{len(RANDOM_CONDITIONS)} random conditions x 2 seed sets; {THREADS} threads, batch {BATCH}")


def seeds_for(cond: dict, scheme: str) -> list:
    if scheme == "position-study":
        return [seed_for(p) for p in positions]
    return [content_seed(s, cond["suite"], cond["corruption"]) for s in image_sha256]


first = list(range(BATCH))
for cond in RANDOM_CONDITIONS:  # check before measuring: position seeds give exactly the study's damage
    study = damaged_batch(crops, positions, first, cond["suite"], cond["corruption"], cond["severity"])
    seeded = damaged_batch_seeded(crops, first, cond["suite"], cond["corruption"], cond["severity"],
                                  seeds_for(cond, "position-study")[:BATCH])
    if not np.array_equal(study, seeded):
        sys.exit(f"STOP: seeded damage differs from the study's for {condition_label(cond)}")
print("check before measuring: PASS (position seeds reproduce the study's damage on the first batch of every "
      "condition)")

runtime = {"name": "onnxruntime", "version": ort.__version__, "execution_provider": "CPUExecutionProvider",
           "threads": THREADS, "spinning": "off"}
cells = []
OUT.mkdir(parents=True, exist_ok=True)
for cond in RANDOM_CONDITIONS:
    correct = {}
    for scheme in ("position-study", "content-v1"):
        started = time.time()
        seeds = seeds_for(cond, scheme)
        scores = {p: [] for p in PRECISIONS}
        for start in range(0, len(positions), BATCH):
            idx = list(range(start, min(start + BATCH, len(positions))))
            batch = normalised(damaged_batch_seeded(crops, idx, cond["suite"], cond["corruption"],
                                                    cond["severity"], [seeds[i] for i in idx]))
            for p in PRECISIONS:
                scores[p].append(sessions[p].run(None, {"images": batch})[0])
        for p in PRECISIONS:
            correct[(p, scheme)] = (np.concatenate(scores[p]).argmax(axis=1) == labels).astype(np.int64)
        print(f"done: {condition_label(cond)}, {scheme}, {duration(time.time() - started)}")
    machine = machine_fingerprint()
    for p in PRECISIONS:
        new, old = correct[(p, "content-v1")], correct[(p, "position-study")]
        diff = paired(new.astype(np.float64), old.astype(np.float64))
        name = f"{MODEL}_{p}_{SPLIT}_{cond['suite']}_{cond['corruption']}_s{cond['severity']}"
        path = OUT / f"{name}.json"
        with written_atomically(path.with_suffix(".npz")) as tmp, tmp.open("wb") as f:
            np.savez_compressed(f, correct_position_study=old, correct_content_v1=new,
                                positions=np.asarray(positions), labels=labels)
        record = make_measurement(
            "diagnostic", {"name": MODEL, "weights": entry["weights"], "licence": entry["licence"]}, p,
            runtime, "laptop", machine,
            {"dataset": DATASET, "split": SPLIT, "n_images": len(positions),
             "licence": DATASETS[DATASET]["licence"]},
            cond,
            {"top1_position_study": metric(float(old.mean()), bootstrap_ci(old.astype(np.float64))),
             "top1_content_v1": metric(float(new.mean()), bootstrap_ci(new.astype(np.float64))),
             "top1_content_v1_minus_position_study": metric(diff["value"], diff["ci95"]),
             "images_content_v1_minus_position_study": metric(diff["count"])},
            {"script": "scripts/52_seed_sensitivity.py", "batch_size": BATCH, "preprocessing": prep,
             "seed_schemes": ["position-study", "content-v1"], "bootstrap": diff["bootstrap"],
             "method": "docs/user_models.md, notes of 7 October 2026 (D18 c)"},
            {"file": path.with_suffix(".npz").name, "sha256": sha256_of(path.with_suffix(".npz"))},
            [{"file": model_files[p].as_posix(), "sha256": sha256_of(model_files[p])}])
        problems = check_record(record)
        if problems:
            sys.exit(f"STOP: the record for {name} fails the schema check: {problems}")
        save_measurement(record, path)
        cells.append({"precision": p, "condition": cond, "label": condition_label(cond),
                      "position_study": float(old.mean()), "content_v1": float(new.mean()),
                      "difference": diff["value"], "ci95": diff["ci95"], "images": diff["count"],
                      "record": path.as_posix()})

largest_end = max(abs(end) for c in cells for end in c["ci95"]) * 100  # in points
m_points = math.ceil(round(largest_end * 10, 9)) / 10  # round() keeps float noise from adding 0.1
adopt = m_points <= STOP_ABOVE_POINTS
print()
print(f"{'build':<22}{'condition':<32}{'position':>10}{'content':>10}{'difference':>12}  "
      "95% interval (points)")
for c in cells:
    print(f"{c['precision']:<22}{c['label']:<32}{c['position_study']:>10.2%}{c['content_v1']:>10.2%}"
          f"{c['difference'] * 100:>+11.2f}  {c['ci95'][0] * 100:+.2f} to {c['ci95'][1] * 100:+.2f} "
          f"({c['images']:+d} images)")
print()
print(f"largest absolute interval end: {largest_end:.3f} points -> M = {m_points:.1f} points")
print(f"M {'is at most' if adopt else 'is ABOVE'} {STOP_ABOVE_POINTS} points: "
      + ("proposed for H's approval (dated note next)" if adopt else "STOP for review; M is not adopted"))
summary = make_record(
    "check", MODEL, " and ".join(PRECISIONS),
    {"settings": {"script": "scripts/52_seed_sensitivity.py", "split": SPLIT, "n_images": len(positions),
                  "conditions": [condition_label(c) for c in RANDOM_CONDITIONS],
                  "rule": "M = largest absolute end of the 12 paired 95% intervals, rounded up to 0.1 point; "
                          f"stop for review if M > {STOP_ABOVE_POINTS}",
                  "method": "docs/user_models.md, notes of 7 October 2026 (D18 c)"},
     "metrics": {"largest_interval_end_points": largest_end, "m_points": m_points,
                 "m_adoptable": adopt, "stop_above_points": STOP_ABOVE_POINTS},
     "raw": {"cells": cells}},
    machine_fingerprint())
save_record(summary, SUMMARY)
print(f"written: {SUMMARY} and {len(cells)} records in {OUT}")
