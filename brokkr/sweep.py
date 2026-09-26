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

import contextlib
import hashlib
import inspect
import json
import shutil
from importlib import metadata
from pathlib import Path

import numpy as np

from brokkr import accuracy
from brokkr.datasets import read_parquet_images
from brokkr.results import written_atomically
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
    # The key file is written last, and every file is written under a temporary name first, so a cache
    # counts as present only if all three files were completed.
    todo = [i for i, p in enumerate(paths) if not all(f.exists() for f in p.values())]
    if todo:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        temp = {i: paths[i]["crops"].with_name(paths[i]["crops"].name + ".tmp") for i in todo}
        crops = {}
        try:
            for i in todo:
                crops[i] = np.lib.format.open_memmap(temp[i], mode="w+", dtype=np.uint8,
                                                     shape=(len(positions), 224, 224, 3))
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
                temp[i].replace(paths[i]["crops"])
                with written_atomically(paths[i]["labels"]) as tmp, tmp.open("wb") as f:
                    np.save(f, labels)
                with written_atomically(paths[i]["key"]) as tmp:
                    tmp.write_text(json.dumps(keys[i], indent=2), encoding="utf-8")
        finally:
            crops.clear()  # release the memory maps so unfinished .tmp files can be removed
            for i in todo:
                with contextlib.suppress(OSError):  # a leftover .tmp file is harmless: it is never read
                    temp[i].unlink(missing_ok=True)
    return paths


def missing_cache_bytes(dataset: str, split: str, positions, files: list, preps: list, cache_dir) -> int:
    """Disk space the caches not yet built will take (for the free-space check before building)."""
    keys = [cache_key(dataset, split, positions, files, prep) for prep in preps]
    missing = [k for k in keys if not all(f.exists() for f in cache_paths(cache_dir, k).values())]
    return len(missing) * len(positions) * (CROP_BYTES + 8)


MIN_FREE_GB = 8.0


def free_gb(folder) -> float:
    return shutil.disk_usage(folder).free / 1e9


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


SLOW_LIMIT = 1.5  # a step taking more than 1.5 times its estimate gets a warning in the log


def estimated_step_seconds(jobs: list, n_images: int, rates: dict, damage_rate: float | None,
                           normalise_rate: float) -> float | None:
    """Expected seconds for one sweep step: damage and normalise n_images once, then run every job.

    rates: images per second for each (model, precision) job; damage_rate: images per second of the
    condition's damage (None for clean). None if any rate is unknown.
    """
    if any(job not in rates for job in jobs):
        return None
    seconds = n_images / normalise_rate + (n_images / damage_rate if damage_rate else 0.0)
    return seconds + sum(n_images / rates[job] for job in jobs)


def slow_warning(actual_seconds: float, estimate_seconds: float | None,
                 limit: float = SLOW_LIMIT) -> str | None:
    """A warning sentence if a step took more than `limit` times its estimate, else None."""
    if not estimate_seconds:
        return None
    ratio = actual_seconds / estimate_seconds
    if ratio <= limit:
        return None
    return (f"WARNING: this step took {ratio:.1f}x its estimate (limit {limit}x): "
            f"{actual_seconds / 60:.1f} min against {estimate_seconds / 60:.1f} min")
