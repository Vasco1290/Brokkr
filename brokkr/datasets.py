"""Test datasets: where they come from, their licences, and how to read them.

Each reader yields (image_bytes, label) pairs one at a time, so a 50,000-image
dataset never has to fit in memory at once.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Every dataset Brokkr uses, with its licence recorded (hard rule 8).
DATASETS = {
    "imagenet-1k-val": {
        "source": "https://huggingface.co/datasets/ILSVRC/imagenet-1k (validation split)",
        "folder": "data/imagenet-1k/data",
        "files": "validation-*.parquet",
        "licence": (
            "ImageNet Terms of Access: non-commercial research and educational use only. "
            "Gated download; each user must accept the terms themselves."
        ),
    },
}


def parquet_files(name: str, root: Path = Path(".")) -> list:
    spec = DATASETS[name]
    files = sorted((root / spec["folder"]).glob(spec["files"]))
    if not files:
        raise FileNotFoundError(f"no files for {name} in {root / spec['folder']}")
    return files


def count_images(files: list) -> int:
    """Total number of images, read from the file headers (fast, no images loaded)."""
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def choose_subset(total: int, n: int | None, seed: int = 0) -> np.ndarray:
    """Pick n image positions out of `total`, the same ones every time for a given seed.

    n=None means use every image.
    """
    if n is None or n >= total:
        return np.arange(total)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=n, replace=False))


def choose_calibration(total: int, n: int, exclude: np.ndarray, seed: int = 1) -> np.ndarray:
    """Pick n image positions for calibration that are NOT in `exclude` (the test images)."""
    available = np.setdiff1d(np.arange(total), exclude)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(available, size=n, replace=False))


# The one place that decides which ImageNet validation images are used for what.
# No image is in two groups. "test" and "int8_calibration" are built exactly as in Stage 1
# (same functions, same seeds), so every Stage 1 result still refers to the same images.
SPLIT_SIZES = {"test": 10_000, "int8_calibration": 512, "conformal_calibration": 5_000, "tuning": 5_000}
SPLIT_SEEDS = {"test": 0, "int8_calibration": 1, "other_splits": 2}


def make_splits(total: int) -> dict:
    """Return {split name: sorted image positions} for a dataset of `total` images.

    test                   the images every accuracy/reliability result is measured on
    int8_calibration       used to build the INT8 model
    conformal_calibration  used to tune conformal prediction sets (Stage 2)
    tuning                 held back for choosing settings in Stage 3 fixes
    """
    test = choose_subset(total, SPLIT_SIZES["test"], seed=SPLIT_SEEDS["test"])
    int8 = choose_calibration(total, SPLIT_SIZES["int8_calibration"], exclude=test,
                              seed=SPLIT_SEEDS["int8_calibration"])
    remaining = np.setdiff1d(np.arange(total), np.concatenate([test, int8]))
    shuffled = np.random.default_rng(SPLIT_SEEDS["other_splits"]).permutation(remaining)
    n_conf, n_tune = SPLIT_SIZES["conformal_calibration"], SPLIT_SIZES["tuning"]
    if len(shuffled) < n_conf + n_tune:
        raise ValueError(f"only {total} images: not enough for all splits")
    return {
        "test": test,
        "int8_calibration": int8,
        "conformal_calibration": np.sort(shuffled[:n_conf]),
        "tuning": np.sort(shuffled[n_conf:n_conf + n_tune]),
    }


def read_parquet_images(files: list, positions: np.ndarray):
    """Yield (image_bytes, label) for the chosen positions, reading one chunk at a time."""
    wanted = set(positions.tolist())
    position = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "label"]):
            images = batch.column("image").to_pylist()  # each is {"bytes": ..., "path": ...}
            labels = batch.column("label").to_pylist()
            for image, label in zip(images, labels, strict=True):
                if position in wanted:
                    yield image["bytes"], label
                position += 1
