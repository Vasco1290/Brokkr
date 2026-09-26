"""A crash mid-write must never leave a half-written file under the real name (brokkr.results)."""

import json

import numpy as np
import pytest

from brokkr import sweep
from brokkr.results import written_atomically


def test_a_finished_write_gets_the_real_name(tmp_path):
    out = tmp_path / "a.json"
    with written_atomically(out) as tmp:
        tmp.write_text("done")
    assert out.read_text() == "done" and not list(tmp_path.glob("*.tmp"))


def test_a_crash_mid_write_leaves_no_file_and_keeps_the_old_one(tmp_path):
    out = tmp_path / "a.json"
    out.write_text("old, complete")
    with pytest.raises(RuntimeError), written_atomically(out) as tmp:
        tmp.write_text("half of the new")
        raise RuntimeError("power cut")
    assert out.read_text() == "old, complete" and not list(tmp_path.glob("*.tmp"))


def test_a_crash_leaves_no_new_file_at_all(tmp_path):
    out = tmp_path / "scores.npz"
    with pytest.raises(RuntimeError), written_atomically(out) as tmp, tmp.open("wb") as f:
        np.savez_compressed(f, logits=np.zeros(3))
        raise RuntimeError("crash after writing, before finishing")
    assert not out.exists()


def test_a_cache_build_that_crashes_leaves_nothing_that_counts_as_a_cache(tmp_path, monkeypatch):
    from test_sweep import PREP_A, fake_parquet  # shared test helpers (tests/ is on pytest's path)
    files = fake_parquet(tmp_path / "a.parquet")
    positions = np.array([0, 1, 2, 3])
    real = sweep.accuracy.resize_and_crop
    calls = {"n": 0}

    def crash_on_third(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("crash mid-cache")
        return real(*args, **kwargs)

    monkeypatch.setattr(sweep.accuracy, "resize_and_crop", crash_on_third)
    with pytest.raises(RuntimeError):
        sweep.build_caches("fake", "test", positions, files, [PREP_A], tmp_path / "cache")
    paths = sweep.cache_paths(tmp_path / "cache", sweep.cache_key("fake", "test", positions, files, PREP_A))
    assert not any(p.exists() for p in paths.values())
    monkeypatch.setattr(sweep.accuracy, "resize_and_crop", real)
    paths = sweep.build_caches("fake", "test", positions, files, [PREP_A], tmp_path / "cache")[0]
    assert all(p.exists() for p in paths.values())                          # a rerun builds it
    assert json.loads(paths["key"].read_text())["n_images"] == 4


def test_missing_cache_bytes_counts_only_caches_not_built(tmp_path):
    from test_sweep import PREP_A, PREP_B, fake_parquet
    files = fake_parquet(tmp_path / "a.parquet")
    positions = np.array([0, 1])
    per_cache = 2 * (sweep.CROP_BYTES + 8)
    both = [PREP_A, PREP_B]
    assert sweep.missing_cache_bytes("fake", "test", positions, files, both, tmp_path) == 2 * per_cache
    sweep.build_caches("fake", "test", positions, files, [PREP_A], tmp_path)
    assert sweep.missing_cache_bytes("fake", "test", positions, files, both, tmp_path) == per_cache
