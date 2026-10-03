"""Write brokkr_edge/model_list.json from torchvision's metadata (needs PyTorch; the result does not).

Usage:  python scripts/38_write_model_list.py [--check]
Writes: brokkr_edge/model_list.json (package data: read by brokkr_edge.model_list without PyTorch)

--check: do not write; exit 1 if the file differs from what torchvision's metadata gives now.
Every value is read from torchvision (brokkr_edge.export.MODELS and each weights' metadata and
transforms); the licence text is brokkr_edge.export's, recorded there on 26 September 2026. The
display names (shown on labels) are typed once, in brokkr_edge.export and brokkr_edge.quantize.
"""

import argparse
import json
import sys

from brokkr_edge.export import FP32_DISPLAY_NAME, MODELS
from brokkr_edge.model_list import MODEL_LIST_FILE
from brokkr_edge.quantize import INT8_METHODS


def model_list() -> dict:
    models = {}
    for name, spec in MODELS.items():
        weights = spec["weights"]
        t = weights.transforms()
        metrics = weights.meta["_metrics"]["ImageNet-1K"]
        models[name] = {
            "publisher": "torchvision",
            "display_name": spec["display_name"],
            "weights": str(weights),
            "licence": spec["licence"],
            "task": "classification",
            "modality": "image",
            "input": {"shape": [3, t.crop_size[0], t.crop_size[0]],
                      "preprocessing": {"resize": t.resize_size[0], "crop": t.crop_size[0],
                                        "interpolation": t.interpolation.value,
                                        "mean": list(t.mean), "std": list(t.std)}},
            "outputs": {"n_classes": len(weights.meta["categories"]), "class_set": "imagenet-1k"},
            "published": {"top1": metrics["acc@1"] / 100, "top5": metrics["acc@5"] / 100,
                          "source": "torchvision weights metadata"},
        }
    precisions = {"fp32": FP32_DISPLAY_NAME} | {
        f"int8_{method}": spec["display_name"] for method, spec in INT8_METHODS.items()
    }
    return {"generated_by": "scripts/38_write_model_list.py", "models": models,
            "precision_display_names": precisions}


parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
text = json.dumps(model_list(), indent=2) + "\n"
if args.check:
    same = MODEL_LIST_FILE.exists() and MODEL_LIST_FILE.read_text(encoding="utf-8") == text
    print("PASS: model_list.json matches torchvision" if same else "FAIL: model_list.json is out of date")
    sys.exit(0 if same else 1)
MODEL_LIST_FILE.write_text(text, encoding="utf-8")
print(f"wrote {MODEL_LIST_FILE} ({len(json.loads(text)['models'])} models)")
