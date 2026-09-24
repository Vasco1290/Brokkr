"""Make picture sheets showing every corruption at every severity, and time each corruption.

Usage:  python scripts/06_corruption_samples.py
Needs:  data/imagenet-1k/
Writes: results/samples/corruptions_<image position>.png (one sheet per sample image)

Each sheet: one row per corruption, columns = severity 0 (clean) to 5 (severe).
Look at them: the damage should look like plausible real-world conditions, and severity 5
should be bad but usually still recognisable to a person.
"""

import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from brokkr.accuracy import open_image, resize_and_crop
from brokkr.datasets import count_images, make_splits, parquet_files, read_parquet_images
from brokkr.shift import CORRUPTIONS, corrupt

TILE, LABEL_W, HEADER_H = 224, 130, 28
out_dir = Path("results/samples")
out_dir.mkdir(parents=True, exist_ok=True)

files = parquet_files("imagenet-1k-val")
test = make_splits(count_images(files))["test"]
positions = test[[0, 2500, 5000]]  # three fixed test images
images = [resize_and_crop(open_image(b)) for b, _ in read_parquet_images(files, positions)]

written = []
for position, image in zip(positions, images, strict=True):
    sheet = Image.new("RGB", (LABEL_W + TILE * 6, HEADER_H + TILE * len(CORRUPTIONS)), "white")
    draw = ImageDraw.Draw(sheet)
    for col in range(6):
        draw.text((LABEL_W + col * TILE + 8, 8), "clean" if col == 0 else f"severity {col}", fill="black")
    for row, name in enumerate(CORRUPTIONS):
        y = HEADER_H + row * TILE
        draw.text((8, y + TILE // 2), name, fill="black")
        for col in range(6):
            tile = corrupt(image, name, col, seed=int(position))
            sheet.paste(Image.fromarray(tile), (LABEL_W + col * TILE, y))
    path = out_dir / f"corruptions_{position}.png"
    sheet.save(path)
    written.append(path)
    print(f"Wrote {path}")

# How long does each corruption take? (Matters for the full sweep over 10,000 images.)
print("\nTime per 224x224 image, severity 3 (mean of 50):")
for name in CORRUPTIONS:
    start = time.perf_counter()
    for i in range(50):
        corrupt(images[0], name, 3, seed=i)
    print(f"  {name:13s} {(time.perf_counter() - start) / 50 * 1000:6.2f} ms")

passed = len(written) == len(positions) and all(p.stat().st_size > 0 for p in written)
print("\nPASS" if passed else "\nFAIL")
sys.exit(0 if passed else 1)
