"""Picture sheet for the pre-4.1 check: do the ImageNet-C corruptions look right? (for viewing only)

Usage:  python scripts/25_imagenet_c_samples.py
Needs:  data/imagenet-1k/, the imagenet-c extra (pyproject.toml)
Writes: results/samples/imagenet_c_check.png

Rows: 5 tuning images (never test images). Columns: clean; the ImageNet-C conditions planned for 4.1
(fog, contrast, defocus blur, Gaussian noise at severities 3 and 5); then Brokkr's own four 4.1
conditions for comparison. Each column is labelled with its suite, e.g. "fog (ImageNet-C) s3".
Same seeds as the sweeps (brokkr.sweep.damaged_batch).
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from brokkr.accuracy import open_image, resize_and_crop
from brokkr.datasets import count_images, make_splits, parquet_files, read_parquet_images
from brokkr.schema import condition, condition_label
from brokkr.sweep import damaged_batch

TILE, LABEL_H = 160, 30
CONDITIONS = ([condition()]
              + [condition(c, "imagenet-c", s) for c in ("fog", "contrast", "defocus_blur", "gaussian_noise")
                 for s in (3, 5)]
              + [condition("fog", "brokkr", 3), condition("darkness", "brokkr", 5),
                 condition("defocus_blur", "brokkr", 3), condition("noise", "brokkr", 3)])

files = parquet_files("imagenet-1k-val")
tuning = make_splits(count_images(files))["tuning"]
positions = tuning[[0, 1000, 2000, 3000, 4000]]
crops = np.stack([resize_and_crop(open_image(b)) for b, _ in read_parquet_images(files, positions)])

sheet = Image.new("RGB", (TILE * len(CONDITIONS), LABEL_H + TILE * len(positions)), "white")
draw = ImageDraw.Draw(sheet)
for col, cond in enumerate(CONDITIONS):
    batch = damaged_batch(crops, positions, list(range(len(positions))), cond["suite"], cond["corruption"],
                          cond["severity"])
    label = condition_label(cond)
    draw.text((col * TILE + 4, 2), label[:26], fill="black")
    draw.text((col * TILE + 4, 15), label[26:], fill="black")
    for row, picture in enumerate(batch):
        sheet.paste(Image.fromarray(picture).resize((TILE, TILE)), (col * TILE, LABEL_H + row * TILE))

out = Path("results/samples/imagenet_c_check.png")
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(f"Saved {out} ({len(positions)} tuning images x {len(CONDITIONS)} conditions). Look at it.")
