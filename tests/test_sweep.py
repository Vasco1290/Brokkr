"""Checks for brokkr.sweep: keyed clean-picture caches and repeatable damaged batches.

Uses tiny fake Parquet files and random pictures (test data, not results).
"""

import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from brokkr import sweep
from brokkr.accuracy import resize_and_crop
from brokkr.shift.corruptions import corrupt as brokkr_corrupt

PREP_A = {"resize": 232, "crop": 224, "interpolation": "bilinear"}
PREP_B = {"resize": 256, "crop": 224, "interpolation": "bicubic"}


def fake_parquet(path, n=6, seed=0):
    """A Parquet file of n random JPEG photos of different shapes, like the ImageNet files."""
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n):
        pixels = (rng.random((240 + 10 * i, 300 - 7 * i, 3)) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(pixels).save(buf, format="JPEG")
        images.append({"bytes": buf.getvalue(), "path": f"{i}.jpg"})
    pq.write_table(pa.table({"image": images, "label": list(range(n))}), path)
    return [path]


def fresh(files, position, prep):
    table = pq.read_table(files[0])
    image = Image.open(io.BytesIO(table["image"][position].as_py()["bytes"])).convert("RGB")
    return resize_and_crop(image, prep["resize"], prep["crop"], prep["interpolation"])


def test_cached_crops_equal_freshly_made_ones(tmp_path):
    files = fake_parquet(tmp_path / "a.parquet")
    positions = np.array([0, 2, 4])
    paths = sweep.build_caches("fake", "test", positions, files, [PREP_A, PREP_B], tmp_path / "cache")
    for prep, p in zip([PREP_A, PREP_B], paths, strict=True):
        crops, labels = sweep.load_cache(p, sweep.cache_key("fake", "test", positions, files, prep))
        assert labels.tolist() == positions.tolist()
        for n, position in enumerate(positions):
            assert np.array_equal(crops[n], fresh(files, position, prep))
    assert not np.array_equal(crops[0], fresh(files, 4, PREP_A))  # the two groups really differ


def test_unsorted_positions_are_refused(tmp_path):
    files = fake_parquet(tmp_path / "a.parquet")
    with pytest.raises(ValueError, match="sorted"):
        sweep.build_caches("fake", "test", np.array([4, 0, 2]), files, [PREP_A], tmp_path)


def test_an_existing_cache_is_reused_only_with_the_same_key(tmp_path):
    files = fake_parquet(tmp_path / "a.parquet")
    positions = np.array([1, 3])
    first = sweep.build_caches("fake", "test", positions, files, [PREP_A], tmp_path)[0]
    made = first["crops"].stat().st_mtime_ns
    again = sweep.build_caches("fake", "test", positions, files, [PREP_A], tmp_path)[0]
    assert again == first and first["crops"].stat().st_mtime_ns == made   # same key: reused
    other = sweep.build_caches("fake", "test", np.array([1, 2]), files, [PREP_A], tmp_path)[0]
    assert other["crops"] != first["crops"]                                # new key: rebuilt


@pytest.mark.parametrize("change", [
    {"split": "tuning"}, {"positions": np.array([1, 2, 4])}, {"prep": PREP_B},
])
def test_the_key_covers_what_made_the_cache(tmp_path, change):
    files = fake_parquet(tmp_path / "a.parquet")
    base = {"split": "test", "positions": np.array([1, 2, 3]), "prep": PREP_A}
    changed = {**base, **change}
    key = sweep.cache_key("fake", base["split"], base["positions"], files, base["prep"])
    other = sweep.cache_key("fake", changed["split"], changed["positions"], files, changed["prep"])
    assert sweep.key_id(key) != sweep.key_id(other)
    assert {"code_sha256", "packages", "source_files", "preprocessing"} <= set(key)


def test_a_cache_with_a_different_stored_key_is_refused(tmp_path):
    files = fake_parquet(tmp_path / "a.parquet")
    positions = np.array([0, 1])
    paths = sweep.build_caches("fake", "test", positions, files, [PREP_A], tmp_path)[0]
    key = sweep.cache_key("fake", "test", positions, files, PREP_A)
    stored = json.loads(paths["key"].read_text())
    stored["packages"]["pillow"] = "0.0"
    paths["key"].write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="rebuild"):
        sweep.load_cache(paths, key)


def pictures(n=4):
    rng = np.random.default_rng(1)
    return (rng.random((n, 224, 224, 3)) * 255).astype(np.uint8)


def test_brokkr_batches_are_repeatable_and_follow_the_stage3_seed_rule():
    crops, positions = pictures(), np.array([17, 5, 9000, 3])
    first = sweep.damaged_batch(crops, positions, [0, 2, 3], "brokkr", "noise", 3)
    second = sweep.damaged_batch(crops, positions, [0, 2, 3], "brokkr", "noise", 3)
    assert np.array_equal(first, second)
    stage3 = np.stack([brokkr_corrupt(crops[i], "noise", 3, seed=int(positions[i])) for i in (0, 2, 3)])
    assert np.array_equal(first, stage3)


def test_imagenet_c_batches_are_repeatable():
    pytest.importorskip("imagecorruptions")
    crops, positions = pictures(), np.array([17, 5, 9000, 3])
    for name in ("fog", "contrast", "defocus_blur", "gaussian_noise"):
        first = sweep.damaged_batch(crops, positions, [1, 3], "imagenet-c", name, 5)
        np.random.seed(999)  # someone else using NumPy's global generator in between
        np.random.random(10)
        assert np.array_equal(first, sweep.damaged_batch(crops, positions, [1, 3], "imagenet-c", name, 5))


def test_clean_batches_are_copies_and_never_change_the_cache():
    crops = pictures()
    before = crops.copy()
    batch = sweep.damaged_batch(crops, np.arange(4), [0, 1], None, "clean", 0)
    batch[:] = 0
    assert np.array_equal(crops, before)


@pytest.mark.skipif(not Path("data/imagenet-1k/data").exists(), reason="ImageNet not on this machine")
def test_real_imagenet_cache_sample_matches_fresh_crops(tmp_path):
    from brokkr.accuracy import open_image
    from brokkr.datasets import count_images, make_splits, parquet_files, read_parquet_images
    files = parquet_files("imagenet-1k-val")
    positions = make_splits(count_images(files))["tuning"][:8]
    paths = sweep.build_caches("imagenet-1k-val", "tuning", positions, files, [PREP_A, PREP_B], tmp_path)
    for prep, p in zip([PREP_A, PREP_B], paths, strict=True):
        crops, _ = sweep.load_cache(p, sweep.cache_key("imagenet-1k-val", "tuning", positions, files, prep))
        for n, (image_bytes, _) in enumerate(read_parquet_images(files, positions)):
            expected = resize_and_crop(open_image(image_bytes), prep["resize"], prep["crop"],
                                       prep["interpolation"])
            assert np.array_equal(crops[n], expected)
