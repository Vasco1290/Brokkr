"""Does switching off ONNX Runtime's thread spinning speed up the sweep without changing any output?

Usage:  python scripts/30_spinning_check.py [--images 512] [--rounds 3]
Reads:  the first N tuning images (never test images)
Writes: results/profile/spinning_check.json (a planning record, not a result)

Copies the 4.1 sweep's pattern: the sweep's first pass (MobileNetV3-Small FP32, ShuffleNetV2 and
MNASNet FP32 + Percentile INT8, 8 threads) takes turns on each batch of 32, and each batch is damaged
(darkness (Brokkr) s5) and normalised first, as in the sweep. Spinning on and off alternate
(on, off, on, off, ...) so that anything else on the laptop affects both alike; every run opens fresh
sessions. Reports images per second per setting and whether every score is bit-identical between them.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from brokkr_edge.accuracy import open_image, resize_and_crop
from brokkr_edge.benchmark import make_session
from brokkr_edge.datasets import count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.export import preprocessing
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.results import make_record, save_record
from brokkr_edge.sweep import damaged_batch, normalised

JOBS = [("mobilenet_v3_small", "fp32"), ("shufflenet_v2_x1_0", "fp32"),
        ("shufflenet_v2_x1_0", "int8_percentile99.99"), ("mnasnet1_0", "fp32"),
        ("mnasnet1_0", "int8_percentile99.99")]
THREADS, BATCH = 8, 32
CONDITION = ("brokkr", "darkness", 5)

parser = argparse.ArgumentParser()
parser.add_argument("--images", type=int, default=512)
parser.add_argument("--rounds", type=int, default=3)
args = parser.parse_args()

prep = preprocessing(JOBS[0][0])  # all five models share this preprocessing (256 / 224 / bilinear)
files = parquet_files("imagenet-1k-val")
positions = make_splits(count_images(files))["tuning"][:args.images]
crops = np.stack([resize_and_crop(open_image(b), prep["resize"], prep["crop"], prep["interpolation"])
                  for b, _ in read_parquet_images(files, positions)])


def one_run(spinning: bool) -> tuple:
    sessions = {job: make_session(Path("models") / f"{job[0]}_{job[1]}.onnx", THREADS, spinning=spinning)
                for job in JOBS}
    outputs = {job: [] for job in JOBS}
    t0 = time.perf_counter()
    for start in range(0, len(positions), BATCH):
        idx = list(range(start, min(start + BATCH, len(positions))))
        batch = normalised(damaged_batch(crops, positions, idx, *CONDITION))
        for job, session in sessions.items():
            outputs[job].append(session.run(None, {"images": batch})[0])
    seconds = time.perf_counter() - t0
    return len(positions) / seconds, {job: np.concatenate(parts) for job, parts in outputs.items()}


rates = {"on": [], "off": []}
first = {}
identical = True
for _ in range(args.rounds):
    for setting in ("on", "off"):
        rate, scores = one_run(spinning=setting == "on")
        rates[setting].append(rate)
        print(f"spinning {setting:<3}: {rate:6.1f} images/s (all five model-precisions per image)",
              flush=True)
        if not first:
            first = scores
        else:
            identical &= all(np.array_equal(first[job], scores[job]) for job in JOBS)

medians = {s: float(np.median(r)) for s, r in rates.items()}
print(f"median: spinning on {medians['on']:.1f}, off {medians['off']:.1f} images/s "
      f"(off / on = {medians['off'] / medians['on']:.2f}); every score bit-identical across all "
      f"{2 * args.rounds} runs: {identical}")
record = make_record("profile", "first sweep pass (5 model-precisions)", "fp32 and int8_percentile99.99", {
    "settings": {"split": "tuning", "n_images": args.images, "threads": THREADS, "batch_size": BATCH,
                 "rounds": args.rounds, "order": "on, off alternating", "condition": "darkness (Brokkr) s5",
                 "jobs": [list(j) for j in JOBS],
                 "note": "Planning check for the 4.1 sweep. Not a result; no accuracy computed."},
    "metrics": {"images_per_second": {s: [round(x, 2) for x in r] for s, r in rates.items()},
                "median_images_per_second": medians, "bit_identical_across_runs": identical},
}, machine_fingerprint())
print(f"Saved {save_record(record, Path('results/profile') / 'spinning_check.json')}")
