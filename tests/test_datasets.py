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


def test_splits_have_the_right_sizes_and_never_overlap():
    from brokkr.datasets import SPLIT_SIZES, make_splits
    splits = make_splits(50_000)
    assert {name: len(pos) for name, pos in splits.items()} == SPLIT_SIZES
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not set(splits[a].tolist()) & set(splits[b].tolist()), f"{a} overlaps {b}"


def test_splits_keep_stage1_images():
    # Stage 1 used choose_subset(seed 0) for test and choose_calibration(seed 1) for INT8.
    # The central splits must be exactly those images, or Stage 1 results would no longer match.
    from brokkr.datasets import choose_calibration, make_splits
    splits = make_splits(50_000)
    stage1_test = choose_subset(50_000, 10_000, seed=0)
    stage1_int8 = choose_calibration(50_000, 512, exclude=stage1_test, seed=1)
    assert np.array_equal(splits["test"], stage1_test)
    assert np.array_equal(splits["int8_calibration"], stage1_int8)


def test_splits_are_reproducible():
    from brokkr.datasets import make_splits
    a, b = make_splits(50_000), make_splits(50_000)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_too_few_images_for_all_splits_is_refused():
    import pytest

    from brokkr.datasets import make_splits
    with pytest.raises(ValueError):
        make_splits(12_000)
