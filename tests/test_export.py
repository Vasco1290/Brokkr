"""Checks for brokkr.export.

Uses random (untrained) weights so the test needs no download. Whether the export is
correct does not depend on which weights are inside.
"""

import onnxruntime as ort
import torch

from brokkr.export import compare_with_pytorch, export_onnx, file_info, load_model


def test_export_matches_pytorch(tmp_path):
    torch.manual_seed(0)
    model = load_model("mobilenet_v3_large", pretrained=False)
    path = export_onnx(model, tmp_path / "model.onnx")

    check = compare_with_pytorch(model, path)
    assert check["max_abs_diff"] < 1e-4
    assert check["top1_agreement"] == 1.0


def test_exported_model_accepts_any_batch_size(tmp_path):
    model = load_model("mobilenet_v3_large", pretrained=False)
    path = export_onnx(model, tmp_path / "model.onnx")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for batch in (1, 3):
        out = session.run(None, {"images": torch.zeros(batch, 3, 224, 224).numpy()})[0]
        assert out.shape == (batch, 1000)


def test_file_info(tmp_path):
    path = export_onnx(load_model("mobilenet_v3_large", pretrained=False), tmp_path / "m.onnx")
    info = file_info(path)
    assert info["size_bytes"] > 0
    assert len(info["sha256"]) == 64
    assert isinstance(info["opset"], int)
