"""Profile each step of the 4.1 pipeline, to plan compute (a speed measurement; no accuracy is read).

Usage:  python scripts/23_profile_pipeline.py --model mobilenet_v3_large
            --onnx fp32=models/mobilenet_v3_large_fp32.onnx --onnx int8_minmax=models/...onnx
Reads:  the first 1,000 images of the tuning split (never test images).
Writes: results/profile/<model>_pipeline_profile.json (a planning record, not a result)

Each step is timed on its own, in images per second:
  - reading the JPEG bytes from Parquet, and decoding them;
  - resize + crop (the model's torchvision settings), and normalising;
  - each damaged condition planned for 4.1 (Brokkr's own and ImageNet-C), on the 224x224 crops;
  - inference for each ONNX file: same session settings as the accuracy runs (4 threads, batch 32,
    not pinned). 20 warm-up batches, then 4 passes over the 1,000 clean images (128 timed batches).
Scores are thrown away: this script never computes or prints an accuracy.
"""

import argparse
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

from brokkr_edge.accuracy import normalize
from brokkr_edge.benchmark import make_session, summarise
from brokkr_edge.datasets import count_images, make_splits, parquet_files, read_parquet_images
from brokkr_edge.fingerprint import machine_fingerprint
from brokkr_edge.results import make_record, save_record
from brokkr_edge.shift.corruptions import corrupt as brokkr_corrupt

DATASET = "imagenet-1k-val"
N_IMAGES, BATCH, THREADS, WARMUP_BATCHES, PASSES = 1000, 32, 4, 20, 4
BROKKR = [("fog", 3), ("darkness", 5), ("defocus_blur", 3), ("noise", 3)]
IMAGENET_C = [(c, s) for c in ("fog", "contrast", "defocus_blur", "gaussian_noise") for s in (3, 5)]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--onnx", action="append", required=True, help="label=path, repeatable")
parser.add_argument("--resize", type=int, default=232)
parser.add_argument("--interpolation", choices=["bilinear", "bicubic"], default="bilinear")
parser.add_argument("--out-suffix", default="", help="added to the output file name")
parser.add_argument("--skip-corruptions", action="store_true", help="time only loading and inference")
args = parser.parse_args()
steps = {}


def timed(label: str, n: int, fn, *fn_args):
    """Run fn(*fn_args) once, record images per second for n images, and return fn's result."""
    t0 = time.perf_counter()
    out = fn(*fn_args)
    seconds = time.perf_counter() - t0
    steps[label] = {"images": n, "seconds": round(seconds, 3), "images_per_second": round(n / seconds, 1)}
    print(f"  {label:<40} {n / seconds:>9.1f} images/s")
    return out


def resize_and_crop(image: Image.Image, size: int, method, crop: int = 224) -> np.ndarray:
    """Same as brokkr_edge.accuracy.resize_and_crop, with the interpolation as a parameter (profile only)."""
    w, h = image.size
    new_w, new_h = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
    image = image.resize((new_w, new_h), method)
    left, top = int(round((new_w - crop) / 2.0)), int(round((new_h - crop) / 2.0))
    return np.asarray(image.crop((left, top, left + crop, top + crop)), dtype=np.uint8)


files = parquet_files(DATASET)
positions = make_splits(count_images(files))["tuning"][:N_IMAGES]
print(f"Profiling {args.model} on {N_IMAGES} tuning images")

raw = timed("read JPEG bytes (Parquet)", N_IMAGES,
            lambda: [image for image, _ in read_parquet_images(files, positions)])
decoded = timed("decode JPEG", N_IMAGES, lambda: [Image.open(io.BytesIO(b)).convert("RGB") for b in raw])
method = Image.BICUBIC if args.interpolation == "bicubic" else Image.BILINEAR
crops = timed(f"resize {args.resize} {args.interpolation} + crop 224", N_IMAGES,
              lambda: np.stack([resize_and_crop(im, args.resize, method) for im in decoded]))
clean = timed("normalise", N_IMAGES, lambda: np.stack([normalize(c) for c in crops]))

not_timed = {}
if not args.skip_corruptions:
    def run_brokkr(name, severity):
        return [brokkr_corrupt(c, name, severity, seed=int(p)) for c, p in zip(crops, positions, strict=True)]

    for name, severity in BROKKR:
        timed(f"{name} (Brokkr) s{severity}", N_IMAGES, run_brokkr, name, severity)
    from brokkr_edge.imagenet_c import damage as imagenet_c_damage  # vendored official code, seeded per image

    def run_imagenet_c(name, severity):
        return [imagenet_c_damage(c, name, severity, seed=int(p))
                for c, p in zip(crops, positions, strict=True)]

    for name, severity in IMAGENET_C:
        try:
            timed(f"{name} (ImageNet-C) s{severity}", N_IMAGES, run_imagenet_c, name, severity)
        except Exception as e:  # reported, never patched here
            not_timed[f"{name} (ImageNet-C) s{severity}"] = f"{type(e).__name__}: {e}"
            print(f"  {name} (ImageNet-C) s{severity}: NOT TIMED ({type(e).__name__}: {e})")

inference = {}
for spec in args.onnx:
    label, path = spec.split("=", 1)
    session = make_session(path, THREADS)
    name = session.get_inputs()[0].name
    batches = [clean[i:i + BATCH] for i in range(0, N_IMAGES, BATCH)]
    for i in range(WARMUP_BATCHES):
        session.run(None, {name: batches[i % len(batches)]})
    batch_ms, pass_rates = [], []
    for _ in range(PASSES):
        t_pass = time.perf_counter()
        for b in batches:
            t0 = time.perf_counter()
            session.run(None, {name: b})
            batch_ms.append((time.perf_counter() - t0) * 1000)
        pass_rates.append(N_IMAGES / (time.perf_counter() - t_pass))
    inference[label] = {"file": path, "pass_images_per_second": [round(r, 1) for r in pass_rates],
                        "images_per_second_median_pass": round(float(np.median(pass_rates)), 1),
                        "batch_ms": {k: round(v, 2) for k, v in summarise(batch_ms).items()},
                        "timed_batches": len(batch_ms), "warmup_batches": WARMUP_BATCHES}
    print(f"  inference {label:<30} {np.median(pass_rates):>9.1f} images/s "
          f"(passes: {', '.join(f'{r:.0f}' for r in pass_rates)})")

options = make_session(args.onnx[0].split("=", 1)[1], THREADS).get_session_options()
record = make_record("profile", args.model, "several", {
    "settings": {"split": "tuning", "n_images": N_IMAGES, "batch_size": BATCH, "num_threads": THREADS,
                 "pinned": False, "resize": args.resize, "interpolation": args.interpolation,
                 "ort_graph_optimization_level": str(options.graph_optimization_level),
                 "ort_inter_op_threads": options.inter_op_num_threads,
                 "ort_execution_mode": str(options.execution_mode),
                 "note": "Planning measurement for task 4.1 compute. Not a result; no accuracy computed."},
    "metrics": {"steps": steps, "inference": inference, "not_timed": not_timed},
}, machine_fingerprint())
out = save_record(record, Path("results/profile") / f"{args.model}_pipeline_profile{args.out_suffix}.json")
print(f"Saved {out}")
