# Vendored copy of imagecorruptions 1.1.2

- **Source:** the `imagecorruptions` 1.1.2 wheel on PyPI (github.com/bethgelab/imagecorruptions), an
  extension of the ImageNet-C corruption code of Hendrycks & Dietterich, packaged for any image size.
- **Licence:** Apache-2.0, `LICENSE` in this folder (copied from the wheel).
- **Copied on 26 September 2026.** SHA-256 of the original files:
  - `corruptions.py`: `adb5944eccfafe0118e777e3300c94420eab486556ca805ea101d3374d130cbb`
  - `__init__.py`: `14c8f4da43f1e7fe2fd0b4b9b3b8d668962aacebd574f888140f22502704129a` (unchanged)

## What was changed, and why

One line, shown exactly in `numpy2_fix.diff`: in `plasma_fractal` (used by `fog`),
`dtype=np.float_` became `dtype=np.float64`. NumPy 2.0 removed the name `np.float_`, which was only an
alias of `np.float64`, so the calculation is unchanged. `tests/test_imagenet_c.py` checks that this
copy's fog equals the installed package's fog (run with `np.float_` temporarily set to `np.float64`)
pixel for pixel.

## Known limits of this copy (not used by Brokkr)

- `glass_blur` and `gaussian_blur` still pass `multichannel=True`, which newer scikit-image removed.
- `frost` loads picture files that were not copied.
