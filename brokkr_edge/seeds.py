"""Damage seeds: which random pattern each image gets (docs/user_models.md, notes of 7 October 2026, D18).

Six of the 12 damaged conditions draw random numbers (fog and noise (Brokkr); fog and Gaussian noise
(ImageNet-C)); the other six do not, so a seed changes nothing for them. Two seed schemes exist:

- "position-study": the image's dataset position (the study, Stages 2-4; brokkr_edge.sweep.seed_for).
- "content-v1": for a user's images. The first 4 bytes, read as a big-endian whole number (0 to 2^32 - 1, the
  range NumPy's global generator accepts, which the ImageNet-C code uses), of the SHA-256 of the UTF-8 text
  "<the image file's SHA-256 in hex>:<suite>/<damage type>". Adding or removing an image changes no other
  image's damage. The severity is left out, so one image gets the same pattern at every severity (the study's
  rule: severities differ only in strength).

damaged_batch_seeded() applies a condition with one given seed per image, choosing the damage code exactly as
brokkr_edge.sweep.damaged_batch does (that function stays as the study's, unchanged).
"""

import hashlib

import numpy as np

from brokkr_edge.shift.corruptions import corrupt as brokkr_corrupt

SEED_SCHEMES = ("position-study", "content-v1")


def content_seed(image_sha256: str, suite: str, damage: str) -> int:
    """The content-v1 seed of one image for one damage type (any severity)."""
    text = f"{image_sha256.lower()}:{suite}/{damage}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def damaged_batch_seeded(crops, indices, suite: str | None, corruption: str, severity: int,
                         seeds) -> np.ndarray:
    """uint8 pictures crops[indices] with the condition applied, seeds[k] for the k-th picture (a copy)."""
    if corruption == "clean":
        return np.array(crops[indices], dtype=np.uint8)
    if suite == "brokkr":
        fn = brokkr_corrupt
    elif suite == "imagenet-c":
        from brokkr_edge.imagenet_c import damage as fn  # optional packages, imported only when used
    else:
        raise ValueError(f"unknown suite {suite!r}")
    return np.stack([fn(np.asarray(crops[i]), corruption, severity, seed=int(s))
                     for i, s in zip(indices, seeds, strict=True)])
