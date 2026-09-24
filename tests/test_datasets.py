"""Checks for brokkr.datasets, using tiny fake Parquet files (no real dataset needed)."""

import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from brokkr.accuracy import open_image
from brokkr.datasets import choose_subset, count_images, read_parquet_images


def fake_parquet(path, labels):
    """Write a Parquet file in the same layout as the Hugging Face ImageNet files."""
    images = []
    for label in labels:
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(label, 0, 0)).save(buf, format="PNG")
        images.append({"bytes": buf.getvalue(), "path": f"{label}.png"})
    pq.write_table(pa.table({"image": images, "label": labels}), path)
    return path


def test_count_and_read_across_files(tmp_path):
    files = [fake_parquet(tmp_path / "a.parquet", [0, 1, 2]),
             fake_parquet(tmp_path / "b.parquet", [3, 4])]
    assert count_images(files) == 5

    samples = list(read_parquet_images(files, np.array([1, 3, 4])))
    assert [label for _, label in samples] == [1, 3, 4]
    # The bytes really are the matching image: red channel was set to the label.
    assert open_image(samples[0][0]).getpixel((0, 0))[0] == 1


def test_subset_is_reproducible_and_sorted():
    a = choose_subset(50_000, 10_000, seed=0)
    b = choose_subset(50_000, 10_000, seed=0)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) == 10_000
    assert np.all(np.diff(a) > 0)


def test_subset_none_means_everything():
    assert choose_subset(7, None).tolist() == list(range(7))
