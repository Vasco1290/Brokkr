"""Checks for brokkr.quantize, using a tiny convolutional model (fast, no dataset needed)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from brokkr.datasets import choose_calibration, choose_subset
from brokkr.export import export_onnx
from brokkr.quantize import INT8_METHODS, ImageBatches, to_fp16, to_int8


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.fc = torch.nn.Linear(8, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x)).mean(dim=(2, 3))
        return self.fc(x)


@pytest.fixture(scope="module")
def fp32_model(tmp_path_factory):
    torch.manual_seed(0)
    return export_onnx(TinyNet().eval(), tmp_path_factory.mktemp("q") / "tiny_fp32.onnx", image_size=32)


def run(path, x):
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]


def test_fp16_is_close_to_fp32(fp32_model, tmp_path):
    fp16 = to_fp16(fp32_model, tmp_path / "tiny_fp16.onnx")
    x = np.random.default_rng(0).standard_normal((4, 3, 32, 32)).astype(np.float32)
    assert run(fp16, x).dtype == np.float32  # inputs/outputs stay FP32
    assert np.abs(run(fp16, x) - run(fp32_model, x)).max() < 1e-2


def test_int8_model_is_quantized_and_runs(fp32_model, tmp_path):
    rng = np.random.default_rng(0)
    calibration = [rng.standard_normal((8, 3, 32, 32)).astype(np.float32) for _ in range(4)]
    int8 = to_int8(fp32_model, tmp_path / "tiny_int8.onnx", calibration)

    ops = {node.op_type for node in onnx.load(str(int8)).graph.node}
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    assert run(int8, calibration[0]).shape == (8, 10)
    assert not (tmp_path / "tiny_int8_prep.onnx").exists()  # temporary file cleaned up


def test_calibration_reader_ends():
    reader = ImageBatches([np.zeros((2, 3, 4, 4))])
    assert reader.get_next()["images"].shape == (2, 3, 4, 4)
    assert reader.get_next() is None


def test_calibration_reader_reads_one_group_at_a_time():
    reader = ImageBatches([np.full((1, 3, 4, 4), i) for i in range(4)])
    assert len(reader) == 4
    reader.set_range(2, 4)
    assert [reader.get_next()["images"][0, 0, 0, 0] for _ in range(2)] == [2, 3]
    assert reader.get_next() is None


@pytest.mark.parametrize("method", list(INT8_METHODS))
def test_every_calibration_method_builds_a_working_model(fp32_model, tmp_path, method):
    rng = np.random.default_rng(0)
    calibration = [rng.standard_normal((8, 3, 32, 32)).astype(np.float32) for _ in range(4)]
    int8 = to_int8(fp32_model, tmp_path / f"tiny_{method}.onnx", calibration, method=method, group_batches=2)
    assert np.abs(run(int8, calibration[0]) - run(fp32_model, calibration[0])).max() < 0.5


def test_grouping_does_not_change_minmax(fp32_model, tmp_path):
    rng = np.random.default_rng(0)
    calibration = [rng.standard_normal((8, 3, 32, 32)).astype(np.float32) for _ in range(4)]
    whole = to_int8(fp32_model, tmp_path / "whole.onnx", calibration)
    grouped = to_int8(fp32_model, tmp_path / "grouped.onnx", calibration, group_batches=2)
    assert np.array_equal(run(whole, calibration[1]), run(grouped, calibration[1]))


def test_calibration_never_overlaps_test_images():
    test = choose_subset(50_000, 10_000, seed=0)
    calib = choose_calibration(50_000, 512, exclude=test, seed=1)
    assert len(calib) == 512
    assert not set(calib.tolist()) & set(test.tolist())
