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


def read_parquet_images(files: list, positions: np.ndarray):
    """Yield (image_bytes, label) for the chosen positions, reading one chunk at a time."""
    wanted = set(positions.tolist())
    position = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "label"]):
            images = batch.column("image").to_pylist()  # each is {"bytes": ..., "path": ...}
            labels = batch.column("label").to_pylist()
            for image, label in zip(images, labels):
                if position in wanted:
                    yield image["bytes"], label
                position += 1
