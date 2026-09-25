"""Five kinds of image damage, each at severities 1 (mild) to 5 (severe).

    fog            haze that washes out contrast, thicker in some places than others
    defocus_blur   the lens is out of focus (every point spreads into a small disc)
    motion_blur    the camera moved during the exposure (every point smears into a line)
    noise          grainy sensor noise, as from a cheap camera or a high ISO setting
    darkness       underexposure: less light reached the sensor

Inspired by the ImageNet-C benchmark (Hendrycks & Dietterich, 2019), but these are our own
short implementations, written to be easy to read. They are not identical to ImageNet-C, so
results are not directly comparable with published ImageNet-C numbers.

All functions take and return a uint8 RGB array of shape (height, width, 3).
Randomness (fog pattern, noise pattern, blur direction) comes from `seed` only, not from the
severity, so the same image gets the same pattern at every severity, just stronger.
"""

import numpy as np
from PIL import Image

SEVERITIES = (1, 2, 3, 4, 5)

# Strength settings per severity 1..5 (index 0 = severity 1).
FOG_THICKNESS = (0.25, 0.40, 0.55, 0.70, 0.85)  # share of the picture replaced by haze, on average
DEFOCUS_RADIUS = (2, 3, 4, 6, 8)  # blur disc radius, in pixels (at 224x224)
MOTION_LENGTH = (5, 9, 13, 19, 27)  # smear length, in pixels
NOISE_STD = (0.04, 0.08, 0.12, 0.18, 0.26)  # noise strength, as a fraction of full brightness
DARKNESS_STOPS = (1, 2, 3, 4, 5)  # exposure reduced by this many photographic stops (2^-stops light)


def to_float(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def to_uint8(pixels: np.ndarray) -> np.ndarray:
    return (np.clip(pixels, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def smooth_noise(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """A cloudy random pattern in 0..1: coarse blobs plus finer detail, like real fog."""
    field = np.zeros((height, width), dtype=np.float32)
    for grid, weight in ((3, 0.5), (6, 0.3), (12, 0.2)):
        coarse = rng.random((grid, grid)).astype(np.float32)
        field += weight * np.asarray(Image.fromarray(coarse, mode="F").resize((width, height),
                                                                               Image.BICUBIC))
    return (field - field.min()) / (field.max() - field.min() + 1e-8)


def convolve(pixels: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blur each colour channel with `kernel` (via FFT), padding edges by reflection."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(pixels, ((ph, ph), (pw, pw), (0, 0)), mode="reflect")
    h, w = padded.shape[:2]
    kernel_f = np.fft.rfft2(kernel, s=(h, w))
    out = np.fft.irfft2(np.fft.rfft2(padded, axes=(0, 1)) * kernel_f[..., None], s=(h, w), axes=(0, 1))
    # The FFT result is shifted by the kernel's top-left corner; take the part aligned with the input.
    return out[kh - 1:kh - 1 + pixels.shape[0], kw - 1:kw - 1 + pixels.shape[1]]


def disc_kernel(radius: int) -> np.ndarray:
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x**2 + y**2 <= radius**2).astype(np.float32)
    return kernel / kernel.sum()


def line_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """A thin line of the given length and direction (odd size so it is centred)."""
    size = length if length % 2 else length + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    centre = size // 2
    angle = np.deg2rad(angle_degrees)
    for t in np.linspace(-(length - 1) / 2, (length - 1) / 2, length * 4):
        x = int(round(centre + t * np.cos(angle)))
        y = int(round(centre + t * np.sin(angle)))
        kernel[y, x] = 1.0
    return kernel / kernel.sum()


def fog(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    pixels = to_float(image)
    thickness = FOG_THICKNESS[severity - 1]
    density = thickness * (0.6 + 0.8 * smooth_noise(*pixels.shape[:2], rng))  # patchy, not uniform
    density = np.clip(density, 0.0, 0.95)[..., None]
    haze = 0.85  # light grey
    return to_uint8(pixels * (1 - density) + haze * density)


def defocus_blur(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    return to_uint8(convolve(to_float(image), disc_kernel(DEFOCUS_RADIUS[severity - 1])))


def motion_blur(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    angle = rng.uniform(0, 180)  # direction of the camera shake
    return to_uint8(convolve(to_float(image), line_kernel(MOTION_LENGTH[severity - 1], angle)))


def noise(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    pattern = rng.standard_normal(image.shape).astype(np.float32)
    return to_uint8(to_float(image) + NOISE_STD[severity - 1] * pattern)


def darkness(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    # Pixel values are gamma-encoded (sRGB). Convert to linear light (about x^2.2), let
    # 2^-stops as much light through, then convert back.
    linear = to_float(image) ** 2.2
    return to_uint8((linear * 2.0 ** -DARKNESS_STOPS[severity - 1]) ** (1 / 2.2))


CORRUPTIONS = {
    "fog": fog,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "noise": noise,
    "darkness": darkness,
}


def corrupt(image: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    """Return a damaged copy of a uint8 RGB image. Severity 0 returns an unchanged copy."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a uint8 RGB array of shape (height, width, 3)")
    if severity == 0:
        return image.copy()
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be 0-5, got {severity}")
    return CORRUPTIONS[name](image, severity, np.random.default_rng(seed))
