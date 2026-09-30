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

# The licence of every torchvision model Brokkr uses, checked 26 September 2026. torchvision's
# metadata gives a licence only for its SWAG weights (non-commercial; never used by Brokkr).
TORCHVISION_LICENCE = {
    "code": "BSD-3-Clause (torchvision)",
    "weights": (
        "Published by torchvision, trained on ImageNet-1k. torchvision states no separate "
        "licence for the weights, and ImageNet's terms of access are for non-commercial "
        "research. Check before redistributing."
    ),
}


def _torchvision(builder, weights) -> dict:
    return {"builder": builder, "weights": weights, "image_size": 224, "licence": TORCHVISION_LICENCE}


# Every model Brokkr can use, with its licence recorded (hard rule 8). The first is the Stage 1-3 model;
# the others are the task 4.1 candidates (docs/hypotheses_stage4.md).
MODELS = {
    "mobilenet_v3_large": _torchvision(models.mobilenet_v3_large,
                                       models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": _torchvision(models.mobilenet_v3_small,
                                       models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v2": _torchvision(models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2),
    "efficientnet_b0": _torchvision(models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "shufflenet_v2_x1_0": _torchvision(models.shufflenet_v2_x1_0,
                                       models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    "mnasnet1_0": _torchvision(models.mnasnet1_0, models.MNASNet1_0_Weights.IMAGENET1K_V1),
    "regnet_y_400mf": _torchvision(models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.IMAGENET1K_V2),
    "resnet18": _torchvision(models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": _torchvision(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "convnext_tiny": _torchvision(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def preprocessing(name: str) -> dict:
    """The model's own resize, crop and interpolation, read from its torchvision weights (not typed)."""
    t = MODELS[name]["weights"].transforms()
    return {"resize": t.resize_size[0], "crop": t.crop_size[0], "interpolation": t.interpolation.value}


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
