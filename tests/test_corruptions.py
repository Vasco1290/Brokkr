"""Checks for brokkr_edge.shift (the image corruptions shared with Argos)."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from brokkr_edge.shift import CORRUPTIONS, SEVERITIES, corrupt
from brokkr_edge.shift.corruptions import convolve, disc_kernel, line_kernel


def sample_image(seed: int = 0) -> np.ndarray:
    """A 224x224 picture with smooth gradients, edges, and fine texture, like a real photo."""
    y, x = np.mgrid[0:224, 0:224] / 223.0
    rng = np.random.default_rng(seed)
    r = 0.2 + 0.6 * x
    g = 0.2 + 0.6 * y
    b = 0.5 + 0.3 * np.sin(12 * x) * np.cos(9 * y)
    image = np.stack([r, g, b], axis=-1)
    image[60:160, 60:160] = [0.9, 0.3, 0.2]  # a solid square with sharp edges
    image += 0.05 * rng.standard_normal(image.shape)  # fine texture
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_shape_and_type_are_kept(name):
    out = corrupt(sample_image(), name, 3)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_severity_zero_changes_nothing(name):
    image = sample_image()
    assert np.array_equal(corrupt(image, name, 0), image)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_stronger_damage_changes_the_image_more(name):
    image = sample_image()
    change = [np.abs(corrupt(image, name, s, seed=7).astype(float) - image).mean() for s in SEVERITIES]
    assert all(a < b for a, b in zip(change, change[1:], strict=False)), f"{name}: {change}"


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_same_seed_same_result(name):
    image = sample_image()
    assert np.array_equal(corrupt(image, name, 4, seed=3), corrupt(image, name, 4, seed=3))


def test_different_seeds_give_different_noise():
    image = sample_image()
    assert not np.array_equal(corrupt(image, "noise", 3, seed=1), corrupt(image, "noise", 3, seed=2))


def test_blur_does_not_shift_the_picture():
    # A single bright dot, blurred with a symmetric kernel, must stay centred on the same spot.
    image = np.zeros((31, 31, 3), dtype=np.float32)
    image[15, 15] = 1.0
    for kernel in (disc_kernel(3), line_kernel(9, 30.0)):
        blurred = convolve(image, kernel)[..., 0]
        ys, xs = np.nonzero(blurred > 1e-6)
        weights = blurred[ys, xs]
        assert np.average(ys, weights=weights) == pytest.approx(15, abs=1e-3)
        assert np.average(xs, weights=weights) == pytest.approx(15, abs=1e-3)
        assert blurred.sum() == pytest.approx(1.0, abs=1e-4)  # blur moves light, never adds any


def test_darkness_makes_the_image_darker_but_keeps_order():
    image = sample_image()
    dark = corrupt(image, "darkness", 3)
    assert dark.mean() < image.mean()
    # Brighter pixels stay brighter (darkness is not scrambling the picture).
    flat, flat_dark = image[..., 0].ravel().astype(int), dark[..., 0].ravel().astype(int)
    order = np.argsort(flat)
    assert np.all(np.diff(flat_dark[order]) >= 0)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        corrupt(sample_image().astype(np.float32), "fog", 1)
    with pytest.raises(ValueError):
        corrupt(sample_image(), "fog", 6)


def test_package_is_self_contained():
    # Shared with Argos: may import only the standard library, numpy, PIL, and its own files.
    allowed = {"numpy", "PIL"}
    for path in Path("brokkr_edge/shift").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert not {"brokkr", "brokkr_edge"} & roots, f"{path} imports from Brokkr"
            assert roots <= allowed | set(sys.stdlib_module_names), f"{path} imports {roots}"
