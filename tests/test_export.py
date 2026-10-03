"""Checks for brokkr_edge.export.

Uses random (untrained) weights so the test needs no download. Whether the export is
correct does not depend on which weights are inside.
"""

import numpy as np
import onnxruntime as ort
import pytest
import torch

from brokkr_edge.export import compare_with_pytorch, export_onnx, file_info, load_model


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export once and share the result between tests (exporting takes several seconds)."""
    torch.manual_seed(0)
    model = load_model("mobilenet_v3_large", pretrained=False)
    path = export_onnx(model, tmp_path_factory.mktemp("export") / "model.onnx")
    return model, path


def test_export_matches_pytorch(exported):
    model, path = exported
    check = compare_with_pytorch(model, path)
    assert check["max_abs_diff"] < 1e-4
    assert check["top1_agreement"] == 1.0


def test_exported_model_accepts_any_batch_size(exported):
    _, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        out = session.run(None, {"images": torch.zeros(batch, 3, 224, 224).numpy()})[0]
        assert out.shape == (batch, 1000)


def test_file_info(exported):
    _, path = exported
    info = file_info(path)
    assert info["size_bytes"] > 0
    assert len(info["sha256"]) == 64
    assert isinstance(info["opset"], int)


def test_every_model_has_a_licence_and_standard_normalisation():
    from brokkr_edge.accuracy import IMAGENET_MEAN, IMAGENET_STD
    from brokkr_edge.export import MODELS
    for name, spec in MODELS.items():
        assert spec["licence"]["code"] and spec["licence"]["weights"], name
        t = spec["weights"].transforms()
        # One normalisation for every model, so a damaged batch can be normalised once per group.
        assert np.allclose(t.mean, IMAGENET_MEAN) and np.allclose(t.std, IMAGENET_STD), name


def test_preprocessing_is_read_from_torchvision():
    from brokkr_edge.export import preprocessing
    assert preprocessing("mobilenet_v3_large") == {"resize": 232, "crop": 224, "interpolation": "bilinear"}
    assert preprocessing("efficientnet_b0")["interpolation"] == "bicubic"


def test_default_preprocessing_is_unchanged_by_the_interpolation_option():
    from PIL import Image

    from brokkr_edge.accuracy import resize_and_crop
    image = Image.fromarray((np.random.default_rng(0).random((300, 260, 3)) * 255).astype(np.uint8))
    assert np.array_equal(resize_and_crop(image), resize_and_crop(image, 232, 224, "bilinear"))
    assert not np.array_equal(resize_and_crop(image), resize_and_crop(image, 232, 224, "bicubic"))
