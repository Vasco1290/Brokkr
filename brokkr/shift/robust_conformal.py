"""Robust conformal prediction: calibrate the 90% threshold on a mix of clean and damaged images.

Plain conformal prediction (conformal.py) is calibrated on clean images, and Stage 2 showed its
90% promise breaking under blur and noise. Here the calibration images are one-third clean and
two-thirds damaged, so the threshold is set with bad conditions in mind. The score and threshold
are exactly the same as in conformal.py; only the calibration images change.

There is still no guarantee for a kind of damage the calibration never saw. That is why the
damage types are held out one at a time and the threshold is tested on the held-out one.

Uses only NumPy, so it can be shared with the Argos project.
"""

import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)


def robust_calibration_plan(n_images: int, allowed: list, clean_fraction: float = 1 / 3,
                            seed: int = 9) -> list:
    """Which version of each calibration image to use: (corruption name or None, severity).

    The images are shuffled once (the shuffle depends only on n_images and seed). The first
    round(n_images x clean_fraction) stay clean; the rest are damaged, the k-th of them with the
    (k mod 20)-th (corruption, severity) pair, so every pair gets as close to the same number of
    images as possible. Each image is used once, in one version.
    """
    order = np.random.default_rng(seed).permutation(n_images)
    n_clean = round(n_images * clean_fraction)
    pairs = [(name, severity) for name in allowed for severity in SEVERITIES]
    plan = [(None, 0)] * n_images
    for k, image in enumerate(order[n_clean:]):
        plan[image] = pairs[k % len(pairs)]
    return plan
