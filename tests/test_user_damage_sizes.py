"""Does every damage condition run at the picture sizes a user's model may have?

Test 5 of the plan of 10 October 2026 (docs/user_models.md, section c), written before the damage code of the
user path. Made-up pictures only.

Found when this test was first run (17 November 2026): Brokkr's own four conditions run at any size tried, but
the ImageNet-C code refuses pictures smaller than 32 x 32 (both sides). So the made-up model of the tests was
raised from 8 x 8 to 32 x 32 (tests/user_made_up.py), and what a user run does with a model below 32 x 32 is a
question for H (docs/user_models.md, note of 17 November 2026).
"""

import warnings

import numpy as np
import pytest
from user_made_up import SIZE

from brokkr_edge.seeds import damaged_batch_seeded
from brokkr_edge.test_run import CONDITIONS

DAMAGED = CONDITIONS[1:]
SIZES = (SIZE, 64, 320)
IMAGENET_C_SMALLEST = 32


def pictures(height, width, n=2, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, height, width, 3), dtype=np.uint8)


def damaged(crops, cond) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the ImageNet-C code warns about NumPy and SciPy details
        return damaged_batch_seeded(crops, list(range(len(crops))), cond["suite"], cond["corruption"],
                                    cond["severity"], seeds=[1, 2][: len(crops)])


def test_there_are_twelve_damaged_conditions():
    assert len(DAMAGED) == 12 and {c["suite"] for c in DAMAGED} == {"brokkr", "imagenet-c"}


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("cond", DAMAGED, ids=lambda c: f"{c['suite']}-{c['corruption']}-s{c['severity']}")
def test_every_condition_runs_and_keeps_the_picture_size(cond, size):
    crops = pictures(size, size)
    out = damaged(crops, cond)
    assert out.shape == crops.shape and out.dtype == np.uint8
    assert not np.array_equal(out, crops)  # the damage was applied


def test_the_made_up_model_is_big_enough_for_every_condition():
    assert SIZE >= IMAGENET_C_SMALLEST


@pytest.mark.parametrize("shape", [(31, 31), (31, 32), (32, 31)])
def test_imagenet_c_refuses_pictures_below_32_pixels(shape):
    """Recorded, not wanted: why the made-up model is 32 x 32, and why H decides what happens below 32."""
    crops = pictures(*shape)
    for cond in (c for c in DAMAGED if c["suite"] == "imagenet-c"):
        with pytest.raises(Exception, match="at least 32 pixels"):
            damaged(crops, cond)


@pytest.mark.parametrize("size", [8, 16, 31])
def test_brokkr_conditions_run_below_32_pixels(size):
    crops = pictures(size, size)
    for cond in (c for c in DAMAGED if c["suite"] == "brokkr"):
        assert damaged(crops, cond).shape == crops.shape
