"""`brokkr-edge test`: run Brokkr's condition set on one model and save a checked record of each run.

For one model it runs the clean test images and the 12 damaged conditions of task 4.1, plus the clean
`conformal_calibration` images (needed later for prediction sets), for the FP32 build and, if its
build record says "usable", the Percentile INT8 build. Each run is saved as a schema-2 accuracy record
(brokkr_edge.schema) with every image's scores beside it.

It needs no PyTorch: what a model expects (resize, crop, licence) comes from brokkr_edge.model_list.
The pictures are made by the code the 4.1 sweep used (brokkr_edge.sweep: the same clean-picture caches,
and each image's damage seeded by its dataset position), with 4.1's settings (8 threads, thread
spinning off, batches of 32), so a model's 4.1 records can be reproduced
(docs/hypotheses_stage4.md, notes of 30 September and 3 October 2026).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from brokkr_edge.accuracy import accuracy_from_logits
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import DATASETS, count_images, make_splits, parquet_files
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.model_list import load_model_list
from brokkr_edge.results import sha256_of, written_atomically
from brokkr_edge.schema import (
    check_build_record,
    condition,
    condition_label,
    make_measurement,
    metric,
    save_measurement,
)
from brokkr_edge.sweep import build_caches, cache_key, damaged_batch, duration, load_cache, normalised

DATASET, BATCH, THREADS = "imagenet-1k-val", 32, 8
PRECISIONS = ("fp32", "int8_percentile99.99")
# The 13 conditions of task 4.1: clean, 4 of Brokkr's own, 8 of ImageNet-C.
CONDITIONS = (
    [condition()]
    + [
        condition("fog", "brokkr", 3),
        condition("darkness", "brokkr", 5),
        condition("defocus_blur", "brokkr", 3),
        condition("noise", "brokkr", 3),
    ]
    + [
        condition(c, "imagenet-c", s)
        for c in ("fog", "contrast", "defocus_blur", "gaussian_noise")
        for s in (3, 5)
    ]
)


def usable_precisions(model: str, models_dir="models") -> list:
    """The builds of `model` that may be tested: the file exists and its build record says "usable"."""
    usable = []
    for precision in PRECISIONS:
        record_path = Path(models_dir) / f"{model}_{precision}.json"
        if not (record_path.exists() and record_path.with_suffix(".onnx").exists()):
            continue
        status, problems = check_build_record(json.loads(record_path.read_text(encoding="utf-8")), set())
        if status == "usable" and not problems:
            usable.append(precision)
    return usable


def record_path(out_dir, model: str, precision: str, split: str, cond: dict) -> Path:
    suite = cond["suite"] or "none"
    name = f"{model}_{precision}_{DATASET}_{split}_{suite}_{cond['corruption']}_s{cond['severity']}.json"
    return Path(out_dir) / name


def complete(path: Path) -> bool:
    """Done = the record and its score file both exist and the score file's checksum matches."""
    if not (path.exists() and path.with_suffix(".npz").exists()):
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return record["arrays"]["sha256"] == sha256_of(path.with_suffix(".npz"))
    except (ValueError, KeyError, TypeError):
        return False


def keep_awake(on: bool) -> None:
    """Ask Windows not to sleep while a run is going (nothing to do on other systems)."""
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def run_model(model: str, out_dir, split: str = "test", limit: int | None = None, threads: int = THREADS,
              models_dir="models", cache_dir="data/cache", log=print) -> list:
    """Run every condition for `model` and return the paths of its records (finished ones are kept).

    split "test" is the real run (13 conditions, then clean conformal_calibration); "tuning" with a
    `limit` is a dry run on the first `limit` tuning images.
    """
    entry = load_model_list()[model]
    prep = {k: entry["input"]["preprocessing"][k] for k in ("resize", "crop", "interpolation")}
    precisions = usable_precisions(model, models_dir)
    if "fp32" not in precisions:
        raise ValueError(f"{model}: no usable FP32 build in {models_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    builds = {p: json.loads((Path(models_dir) / f"{model}_{p}.json").read_text(encoding="utf-8"))
              for p in precisions}
    sessions = {p: make_session(Path(models_dir) / f"{model}_{p}.onnx", threads, spinning=False)
                for p in precisions}
    runtime = {"name": "onnxruntime", "version": ort.__version__,
               "execution_provider": "CPUExecutionProvider", "threads": threads, "spinning": "off"}
    files = parquet_files(DATASET)
    splits = make_splits(count_images(files))
    jobs = [(split, CONDITIONS)] + ([("conformal_calibration", [condition()])] if split == "test" else [])
    log(f"brokkr-edge test: {model}, builds {', '.join(precisions)}; split {split}, limit {limit}; "
        f"{threads} threads, spinning off, batch {BATCH}; records in {out_dir}")

    paths = []
    keep_awake(True)
    try:
        for job_split, conditions in jobs:
            positions = splits[job_split][:limit] if limit else splits[job_split]
            cache = build_caches(DATASET, job_split, positions, files, [prep], cache_dir)[0]
            crops, labels = load_cache(cache, cache_key(DATASET, job_split, positions, files, prep))
            for cond in conditions:
                here = {p: record_path(out_dir, model, p, job_split, cond) for p in precisions}
                paths += list(here.values())
                todo = [p for p in precisions if not complete(here[p])]
                if not todo:
                    log(f"skip (complete): {job_split} {condition_label(cond)}")
                    continue
                started = time.time()
                scores = {p: [] for p in todo}
                for start in range(0, len(positions), BATCH):
                    idx = list(range(start, min(start + BATCH, len(positions))))
                    batch = normalised(damaged_batch(crops, positions, idx, cond["suite"],
                                                     cond["corruption"], cond["severity"]))
                    for p in todo:
                        scores[p].append(sessions[p].run(None, {"images": batch})[0])
                machine = machine_fingerprint()
                for p in todo:
                    logits = np.concatenate(scores[p]).astype(np.float32)
                    path = here[p]
                    with written_atomically(path.with_suffix(".npz")) as tmp, tmp.open("wb") as f:
                        np.savez_compressed(f, logits=logits, labels=labels, positions=np.asarray(positions))
                    acc = accuracy_from_logits(logits, labels)["metrics"]
                    record = make_measurement(
                        "accuracy",
                        {"name": model, "weights": entry["weights"], "licence": entry["licence"]},
                        p, runtime, "laptop", machine,
                        {"dataset": DATASET, "split": job_split, "n_images": len(positions),
                         "licence": DATASETS[DATASET]["licence"]},
                        cond,
                        {"top1": metric(acc["top1"], acc["top1_ci95"]),
                         "top5": metric(acc["top5"], acc["top5_ci95"]),
                         "top1_tied_images": metric(acc["top1_tied_images"])},
                        {"script": "brokkr-edge test", "batch_size": BATCH, "preprocessing": prep,
                         "damage_seed": "dataset position of each image", "bootstrap_resamples": 1000,
                         "seed": 0, "top1_range_over_tie_breaks": acc["top1_range_over_tie_breaks"],
                         "cache": cache["crops"].name},
                        {"file": path.with_suffix(".npz").name,
                         "sha256": sha256_of(path.with_suffix(".npz"))},
                        [{"file": f"models/{model}_{p}.onnx", "sha256": builds[p]["file"]["sha256"]}],
                    )
                    save_measurement(record, path)
                log(f"done: {job_split} {condition_label(cond)}, {len(todo)} builds, "
                    f"{duration(time.time() - started)}")
    finally:
        keep_awake(False)
    return paths


def compare_scores(old: dict, new: dict, repeatable: bool) -> dict:
    """Does a new run reproduce an older one? Each is {"logits", "labels", "positions"} of one record.

    The reproduction rule (docs/hypotheses_stage4.md, 30 September 2026): the same images in the same
    order, and then, if inference is repeatable on this machine, identical top-1 predictions on every
    image; if not, the number of correct top-1 answers within 0.1 points (1 image per 1,000).
    """
    same_images = bool(np.array_equal(old["positions"], new["positions"])
                       and np.array_equal(old["labels"], new["labels"]))
    row = {"images": int(len(new["labels"])), "same_images": same_images, "reproduced": False}
    if not same_images:
        return row
    old_top1, new_top1 = old["logits"].argmax(axis=1), new["logits"].argmax(axis=1)  # ties: lower class
    row["top1_differs_on"] = int((old_top1 != new_top1).sum())
    row["correct_then"] = int((old_top1 == old["labels"]).sum())
    row["correct_now"] = int((new_top1 == new["labels"]).sum())
    row["scores_identical"] = bool(np.array_equal(old["logits"], new["logits"]))
    row["largest_score_difference"] = float(np.abs(old["logits"] - new["logits"]).max())
    if repeatable:
        row["reproduced"] = row["top1_differs_on"] == 0
    else:
        row["reproduced"] = abs(row["correct_now"] - row["correct_then"]) <= row["images"] // 1000
    return row
