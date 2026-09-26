"""Checks for the vendored ImageNet-C code and brokkr.imagenet_c (pictures are random test data)."""

import numpy as np
import pytest

# Skipped where the optional ImageNet-C packages are not installed (e.g. the basic CI job).
installed = pytest.importorskip("imagecorruptions")
vendored_corrupt = pytest.importorskip("brokkr.third_party.imagecorruptions").corrupt
imagenet_c = pytest.importorskip("brokkr.imagenet_c")
CORRUPTIONS, damage = imagenet_c.CORRUPTIONS, imagenet_c.damage


def pictures(n=3, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.random((224, 224, 3)) * 255).astype(np.uint8) for _ in range(n)]


def run(corrupt_fn, picture, name, severity, seed):
    np.random.seed(seed)
    return corrupt_fn(picture, corruption_name=name, severity=severity)


@pytest.mark.parametrize("severity", [3, 5])
def test_vendored_fog_equals_the_installed_package(monkeypatch, severity):
    # The installed package's fog only runs on NumPy 2 if np.float_ exists again, as np.float64.
    # If the vendored copy equals it pixel for pixel, the one-line fix changed nothing else.
    monkeypatch.setattr(np, "float_", np.float64, raising=False)
    for i, picture in enumerate(pictures()):
        expected = run(installed.corrupt, picture, "fog", severity, seed=i)
        assert np.array_equal(run(vendored_corrupt, picture, "fog", severity, seed=i), expected)


@pytest.mark.parametrize("name", ["contrast", "defocus_blur", "gaussian_noise"])
def test_other_vendored_corruptions_equal_the_installed_package(name):
    for i, picture in enumerate(pictures()):
        for severity in (3, 5):
            assert np.array_equal(run(vendored_corrupt, picture, name, severity, seed=i),
                                  run(installed.corrupt, picture, name, severity, seed=i))


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_same_seed_gives_identical_pixels(name):
    picture = pictures(1)[0]
    first = damage(picture, name, 3, seed=7)
    np.random.seed(123)            # disturb the global generator in between
    np.random.normal(size=1000)
    assert np.array_equal(damage(picture, name, 3, seed=7), first)
    assert first.dtype == np.uint8 and first.shape == picture.shape


def test_damage_puts_the_global_generator_back():
    np.random.seed(5)
    expected = np.random.random(3)
    np.random.seed(5)
    damage(pictures(1)[0], "gaussian_noise", 3, seed=99)
    assert np.array_equal(np.random.random(3), expected)


def test_different_seeds_give_different_noise():
    picture = pictures(1)[0]
    assert not np.array_equal(damage(picture, "gaussian_noise", 3, seed=1),
                              damage(picture, "gaussian_noise", 3, seed=2))


def test_unknown_corruption_is_refused():
    with pytest.raises(ValueError):
        damage(pictures(1)[0], "frost", 3, seed=0)
