"""Would more threads speed up the 4.1 sweep without changing any output? (planning check, not a result)

Usage:  python scripts/29_thread_check.py --model convnext_tiny [--threads 4 8] [--images 64]
Reads:  the first N tuning images (never test images), the model's own preprocessing
Writes: results/profile/<model>_thread_check.json

For FP32 and Percentile INT8: the same batches are run with each thread count, in alternating passes
(4, 8, 4, 8, ...) so that the laptop warming up affects both alike. Reports images per second (median
of 3 passes, after 2 warm-up batches) and whether the scores are bit-identical between thread counts.
The sweep changes its thread count only if every output is bit-identical.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from brokkr.accuracy import open_image, preprocess
from brokkr.benchmark import make_session
from brokkr.datasets import count_images, make_splits, parquet_files, read_parquet_images
from brokkr.export import preprocessing
from brokkr.fingerprint import machine_fingerprint
from brokkr.results import make_record, save_record

BATCH, PASSES, WARMUP_BATCHES = 32, 3, 2

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--threads", type=int, nargs="+", default=[4, 8])
parser.add_argument("--images", type=int, default=64)
args = parser.parse_args()
prep = preprocessing(args.model)

files = parquet_files("imagenet-1k-val")
positions = make_splits(count_images(files))["tuning"][:args.images]
images = np.stack([preprocess(open_image(b), prep["resize"], prep["crop"], prep["interpolation"])
                   for b, _ in read_parquet_images(files, positions)])
batches = [images[i:i + BATCH] for i in range(0, len(images), BATCH)]

report = {}
for precision in ("fp32", "int8_percentile99.99"):
    path = Path("models") / f"{args.model}_{precision}.onnx"
    sessions = {n: make_session(path, n) for n in args.threads}
    for session in sessions.values():
        for i in range(WARMUP_BATCHES):
            session.run(None, {"images": batches[i % len(batches)]})
    rates, scores = {n: [] for n in args.threads}, {}
    for _ in range(PASSES):
        for n, session in sessions.items():
            t0 = time.perf_counter()
            out = np.concatenate([session.run(None, {"images": b})[0] for b in batches])
            rates[n].append(len(images) / (time.perf_counter() - t0))
            if n in scores and not np.array_equal(scores[n], out):
                raise SystemExit(f"STOP: {precision} with {n} threads is not repeatable between passes")
            scores[n] = out
    base = args.threads[0]
    identical = {n: bool(np.array_equal(scores[base], scores[n])) for n in args.threads}
    largest = {n: float(np.abs(scores[base] - scores[n]).max()) for n in args.threads}
    same_top = {n: float(np.mean(scores[base].argmax(1) == scores[n].argmax(1))) for n in args.threads}
    medians = {n: round(float(np.median(r)), 2) for n, r in rates.items()}
    report[precision] = {"images_per_second_median": medians,
                         "passes": {n: [round(x, 2) for x in r] for n, r in rates.items()},
                         f"bit_identical_to_{base}_threads": identical, "largest_score_difference": largest,
                         "same_top_answer_share": same_top}
    for n in args.threads:
        print(f"{args.model} {precision:<21} {n} threads: {np.median(rates[n]):6.2f} images/s "
              f"(passes {', '.join(f'{x:.2f}' for x in rates[n])}); bit-identical to {base} threads: "
              f"{identical[n]}; largest score difference {largest[n]:.3g}; same top answer {same_top[n]:.1%}")

record = make_record("profile", args.model, "fp32 and int8_percentile99.99", {
    "settings": {"split": "tuning", "n_images": args.images, "batch_size": BATCH, "threads": args.threads,
                 "passes": PASSES, "warmup_batches": WARMUP_BATCHES, "pinned": False,
                 "note": "Planning check for the 4.1 thread count. Not a result; no accuracy computed."},
    "metrics": report,
}, machine_fingerprint())
print(f"Saved {save_record(record, Path('results/profile') / f'{args.model}_thread_check.json')}")
