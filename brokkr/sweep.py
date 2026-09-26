"""Clean-picture caches and damaged batches for sweeps over many models (task 4.1 onward).

Models with the same preprocessing (resize, crop, interpolation) see the same 224x224 pictures, so
they share one cache of clean pictures, and each damaged batch is made once and given to every model
and precision in that group.

- **Cache keys.** A cache is keyed by everything that made it: dataset and split, the exact image
  positions, the source files (names and sizes), the preprocessing settings, the source code of the
  functions that decode and resize, and the NumPy and Pillow versions. The key's hash is part of the
  file name, so a cache made any other way is never found, let alone reused: a stale key means a
  rebuild.
- **One pass.** Reading JPEG bytes from Parquet is the slowest step (profiled 26 September 2026), so
  each image is read and decoded once and resized for every preprocessing group.
- **Damaged batches are regenerated, not cached** (a cached copy would take 1.5 GB per condition per
  group). Each image's damage uses a fixed seed, its dataset position, for every suite, corruption and
  severity. This is Stage 2's rule: the same image gets the same random pattern at every severity, and
  Stage 3's damaged pictures are reproduced exactly (tested).
"""

import hashlib
import inspect
import json
from importlib import metadata
from pathlib import Path

import numpy as np

from brokkr import accuracy
from brokkr.datasets import read_parquet_images
from brokkr.shift.corruptions import corrupt as brokkr_corrupt

CROP_BYTES = 224 * 224 * 3  # one cached picture, uint8


def seed_for(position: int) -> int:
    """The damage seed of one image: its dataset position, for every condition and severity."""
    return int(position)


def cache_key(dataset: str, split: str, positions: np.ndarray, files: list, prep: dict) -> dict:
    """Everything that determines a cache's contents."""
    code = inspect.getsource(accuracy.open_image) + inspect.getsource(accuracy.resize_and_crop)
    return {
        "dataset": dataset,
        "split": split,
        "n_images": int(len(positions)),
        "positions_sha256": hashlib.sha256(np.asarray(positions, dtype=np.int64).tobytes()).hexdigest(),
        "source_files": [{"name": Path(f).name, "bytes": Path(f).stat().st_size} for f in files],
        "preprocessing": {"resize": prep["resize"], "crop": prep["crop"],
                          "interpolation": prep["interpolation"]},
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "packages": {p: metadata.version(p) for p in ("numpy", "pillow")},
    }


def key_id(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def cache_paths(cache_dir, key: dict) -> dict:
    p = key["preprocessing"]
    stem = f"{key['dataset']}_{key['split']}_{p['resize']}-{p['crop']}-{p['interpolation']}_{key_id(key)}"
    cache_dir = Path(cache_dir)
    return {"crops": cache_dir / f"{stem}_crops.npy", "labels": cache_dir / f"{stem}_labels.npy",
            "key": cache_dir / f"{stem}_key.json"}


def build_caches(dataset: str, split: str, positions: np.ndarray, files: list, preps: list,
                 cache_dir) -> list:
    """Make the clean-picture cache of every preprocessing in `preps` that is missing, in one pass.

    Returns the cache paths, in the order of `preps`. Existing caches with the same key are kept.
    """
    if not np.all(np.diff(positions) > 0):
        # read_parquet_images returns images in file order, so any other order would pair the wrong
        # image with its label and its damage seed.
        raise ValueError("positions must be sorted and unique")
    keys = [cache_key(dataset, split, positions, files, prep) for prep in preps]
    paths = [cache_paths(cache_dir, key) for key in keys]
    todo = [i for i, p in enumerate(paths) if not all(f.exists() for f in p.values())]
    if todo:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        temp = {i: paths[i]["crops"].with_suffix(".tmp.npy") for i in todo}
        crops = {i: np.lib.format.open_memmap(temp[i], mode="w+", dtype=np.uint8,
                                              shape=(len(positions), 224, 224, 3)) for i in todo}
        labels = np.empty(len(positions), dtype=np.int64)
        for n, (image_bytes, label) in enumerate(read_parquet_images(files, positions)):
            image = accuracy.open_image(image_bytes).convert("RGB")  # decoded once for every group
            labels[n] = label
            for i in todo:
                p = keys[i]["preprocessing"]
                crops[i][n] = accuracy.resize_and_crop(image, p["resize"], p["crop"], p["interpolation"])
        for i in todo:
            crops[i].flush()
            del crops[i]
            temp[i].rename(paths[i]["crops"])  # only a complete cache gets its real name
            np.save(paths[i]["labels"], labels)
            paths[i]["key"].write_text(json.dumps(keys[i], indent=2))
    return paths


def load_cache(paths: dict, expected_key: dict) -> tuple:
    """(crops as a read-only memory map, labels). Refuses a cache whose stored key differs."""
    stored = json.loads(paths["key"].read_text())
    if stored != expected_key:
        raise ValueError(f"{paths['crops']} was made differently from what is asked for; rebuild it")
    return np.load(paths["crops"], mmap_mode="r"), np.load(paths["labels"])


def damaged_batch(crops, positions, indices, suite: str | None, corruption: str,
                  severity: int) -> np.ndarray:
    """uint8 pictures crops[indices] with the condition applied (a copy; the cache is never changed)."""
    if corruption == "clean":
        return np.array(crops[indices], dtype=np.uint8)
    if suite == "brokkr":
        fn = brokkr_corrupt
    elif suite == "imagenet-c":
        from brokkr.imagenet_c import damage as fn  # optional packages, imported only when used
    else:
        raise ValueError(f"unknown suite {suite!r}")
    return np.stack([fn(np.asarray(crops[i]), corruption, severity, seed=seed_for(positions[i]))
                     for i in indices])


def normalised(pictures: np.ndarray) -> np.ndarray:
    """uint8 (N, 224, 224, 3) -> the float (N, 3, 224, 224) batch every torchvision model here expects."""
    return np.stack([accuracy.normalize(p) for p in pictures])
