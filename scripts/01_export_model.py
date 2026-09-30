"""Export a pretrained model to ONNX FP32 and check it matches PyTorch.

Usage:  python scripts/01_export_model.py [model_name]
Writes: models/<name>_fp32.onnx and models/<name>_fp32.json (details of the export)
"""

import json
import sys
from pathlib import Path

from brokkr_edge.export import MODELS, compare_with_pytorch, export_onnx, file_info, load_model
from brokkr_edge.fingerprint import machine_fingerprint

MAX_ABS_DIFF = 1e-4  # PyTorch and ONNX Runtime outputs must agree to within this

name = sys.argv[1] if len(sys.argv) > 1 else "mobilenet_v3_large"
spec = MODELS[name]
onnx_path = Path("models") / f"{name}_fp32.onnx"

model = load_model(name)
export_onnx(model, onnx_path, spec["image_size"])
check = compare_with_pytorch(model, onnx_path, spec["image_size"])

record = {
    "model": name,
    "precision": "fp32",
    "weights": str(spec["weights"]),
    "licence": spec["licence"],
    "file": {"path": str(onnx_path), **file_info(onnx_path)},
    "pytorch_vs_onnx": check,
    "machine": machine_fingerprint(),
}
onnx_path.with_suffix(".json").write_text(json.dumps(record, indent=2))

print(f"Exported {name} -> {onnx_path} ({record['file']['size_bytes'] / 1e6:.1f} MB)")
print(f"Max difference vs PyTorch: {check['max_abs_diff']:.2e}  (limit {MAX_ABS_DIFF:.0e})")
print(f"Same top-1 class: {check['top1_agreement']:.0%} of {check['batch_size']} inputs")
passed = check["max_abs_diff"] < MAX_ABS_DIFF and check["top1_agreement"] == 1.0
print("PASS" if passed else "FAIL")
sys.exit(0 if passed else 1)
