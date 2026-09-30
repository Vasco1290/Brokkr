"""The models Brokkr labels, readable WITHOUT PyTorch (for testing and labelling on small devices).

The list is a JSON file generated from torchvision's own metadata by scripts/38_write_model_list.py,
so nothing in it is typed by hand; tests/test_model_list.py regenerates it and fails if it no longer
matches torchvision. brokkr_edge.export keeps the PyTorch side (building and exporting the models).

Each entry: the torchvision weights name, the licence (hard rule 8), the preprocessing those weights
expect (resize, crop, interpolation, mean, std), the number of classes, and torchvision's published
ImageNet-1k top-1 and top-5 (used for the FP32 sanity check).
"""

import json
from pathlib import Path

MODEL_LIST_FILE = Path(__file__).with_name("model_list.json")


def load_model_list() -> dict:
    """{model name: entry}, in the file's order."""
    return json.loads(MODEL_LIST_FILE.read_text(encoding="utf-8"))["models"]
