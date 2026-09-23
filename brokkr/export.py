"""Load a pretrained vision model and convert it to an ONNX file.

ONNX is a standard file format for neural networks. Once a model is an .onnx file,
ONNX Runtime can run it on laptops, Raspberry Pis, and many other devices without PyTorch.
"""

import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torchvision import models

# Every model Brokkr can use, with its licence recorded (hard rule 8).
MODELS = {
    "mobilenet_v3_large": {
        "builder": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
        "image_size": 224,
        "licence": {
            "code": "BSD-3-Clause (torchvision)",
            "weights": (
                "Published by torchvision, trained on ImageNet-1k. torchvision states no separate "
                "licence for the weights, and ImageNet's terms of access are for non-commercial "
                "research. Check before redistributing."
            ),
        },
    },
}


def load_model(name: str, pretrained: bool = True) -> torch.nn.Module:
    """Build a model from MODELS in evaluation mode. pretrained=False gives random weights (for tests)."""
    spec = MODELS[name]
    weights = spec["weights"] if pretrained else None
    model = spec["builder"](weights=weights)
    return model.eval()


def export_onnx(model: torch.nn.Module, path: Path, image_size: int = 224) -> Path:
    """Save the model as a single .onnx file that accepts any batch size."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(2, 3, image_size, image_size)
    torch.onnx.export(
        model,
        (example,),
        str(path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},  # batch size can vary
        external_data=False,  # keep weights inside the one .onnx file
    )
    onnx.checker.check_model(str(path))
    return path


def compare_with_pytorch(model: torch.nn.Module, onnx_path: Path, image_size: int = 224,
                         batch_size: int = 8, seed: int = 0) -> dict:
    """Run the same random inputs through PyTorch and ONNX Runtime and compare the outputs.

    If the export is correct, the outputs should be nearly identical and pick the same top class.
    """
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((batch_size, 3, image_size, image_size)).astype(np.float32)

    with torch.no_grad():
        expected = model(torch.from_numpy(inputs)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": inputs})[0]

    return {
        "batch_size": batch_size,
        "seed": seed,
        "max_abs_diff": float(np.max(np.abs(expected - actual))),
        "top1_agreement": float(np.mean(expected.argmax(axis=1) == actual.argmax(axis=1))),
    }


def file_info(path: Path) -> dict:
    """Size, SHA-256 checksum, and ONNX opset of a model file."""
    data = Path(path).read_bytes()
    opsets = {o.domain or "ai.onnx": o.version for o in onnx.load(str(path)).opset_import}
    return {
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "opset": opsets.get("ai.onnx"),
    }
