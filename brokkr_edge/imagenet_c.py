"""ImageNet-C corruptions with a fixed seed per image.

The corruption code is the official ImageNet-C code (the imagecorruptions package, vendored in
brokkr_edge/third_party/imagecorruptions with a one-line NumPy 2 fix). That code draws its random numbers
from NumPy's global generator (np.random.normal and friends). A result must never depend on whatever
state that generator happens to be in, so damage() seeds it for this one call from the given seed and
then puts the previous state back. The same (picture, corruption, severity, seed) therefore always
gives the same pixels, whatever ran before (tested).

Not part of brokkr_edge/shift: that folder is shared with Argos and depends only on NumPy and Pillow,
while this code needs scikit-image, SciPy and OpenCV (the optional extra "imagenet-c").
"""

import numpy as np

from brokkr_edge.third_party.imagecorruptions import corrupt

CORRUPTIONS = ("fog", "contrast", "defocus_blur", "gaussian_noise")


def damage(picture: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    """Return an ImageNet-C damaged copy of a uint8 (height, width, 3) picture."""
    if name not in CORRUPTIONS:
        raise ValueError(f"{name!r} is not one of the ImageNet-C corruptions Brokkr uses: {CORRUPTIONS}")
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        return corrupt(picture, corruption_name=name, severity=severity)
    finally:
        np.random.set_state(saved)
