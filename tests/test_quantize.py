"""Checks for brokkr_edge.quantize, using a tiny convolutional model (fast, no dataset needed)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from brokkr_edge.datasets import choose_calibration, choose_subset
from brokkr_edge.export import export_onnx
from brokkr_edge.quantize import (
    INT8_METHODS,
    ImageBatches,
    damaged_calibration_plan,
    to_fp16,
    to_int8,
    weight_quantization,
)


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


def test_int8_weights_are_per_channel_in_the_file(fp32_model, tmp_path):
    rng = np.random.default_rng(0)
    calibration = [rng.standard_normal((8, 3, 32, 32)).astype(np.float32) for _ in range(2)]
    int8 = to_int8(fp32_model, tmp_path / "tiny_int8.onnx", calibration)
    assert weight_quantization(int8) == {"per_channel": 2, "per_tensor": 0}  # the conv and the linear layer


def test_damaged_calibration_plan():
    allowed = ["fog", "noise", "darkness", "motion_blur"]
    plan = damaged_calibration_plan(512, allowed)
    damaged = [(name, severity) for name, severity in plan if name is not None]
    assert len(damaged) == 256  # exactly half
    assert {name for name, _ in damaged} <= set(allowed)
    assert {severity for _, severity in damaged} == {1, 2, 3, 4, 5}
    assert all(severity == 0 for name, severity in plan if name is None)
    assert damaged_calibration_plan(512, allowed) == plan  # reproducible
    # Another leave-one-out list: same images damaged at the same severities.
    other = damaged_calibration_plan(512, ["fog", "noise", "darkness", "defocus_blur"])
    assert [(n is None, s) for n, s in other] == [(n is None, s) for n, s in plan]


def test_unrounded_output_leaves_the_final_layer_in_float(fp32_model, tmp_path):
    rng = np.random.default_rng(0)
    calibration = [rng.standard_normal((8, 3, 32, 32)).astype(np.float32) for _ in range(2)]

    def gemm_output_is_quantized(path):
        graph = onnx.load(str(path)).graph
        gemm_out = next(n for n in graph.node if n.op_type == "Gemm").output[0]
        return any(n.op_type == "QuantizeLinear" and n.input[0] == gemm_out for n in graph.node)

    rounded = to_int8(fp32_model, tmp_path / "rounded.onnx", calibration)
    unrounded = to_int8(fp32_model, tmp_path / "unrounded.onnx", calibration, unrounded_output_ops=["Gemm"])
    assert gemm_output_is_quantized(rounded) and not gemm_output_is_quantized(unrounded)
    assert weight_quantization(unrounded) == {"per_channel": 2, "per_tensor": 0}  # weights still int8


def test_keep_float_outputs_leaves_those_nodes_unquantized(fp32_model, tmp_path):
    rng = np.random.default_rng(0)
    calibration = [rng.standard_normal((8, 3, 32, 32)).astype(np.float32) for _ in range(2)]
    relu_out = next(n for n in onnx.load(str(fp32_model)).graph.node if n.op_type == "Relu").output[0]

    def quantized(path, tensor):
        graph = onnx.load(str(path)).graph
        return any(n.op_type == "QuantizeLinear" and n.input[0] == tensor for n in graph.node)

    report = {}
    normal = to_int8(fp32_model, tmp_path / "normal.onnx", calibration)
    kept = to_int8(fp32_model, tmp_path / "kept.onnx", calibration, keep_float_outputs=[relu_out],
                   report=report)
    assert quantized(normal, relu_out) and not quantized(kept, relu_out)
    assert len(report["nodes_kept_float"]) == 1 and run(kept, calibration[0]).shape == (8, 10)
    with pytest.raises(ValueError, match="not found"):
        to_int8(fp32_model, tmp_path / "bad.onnx", calibration, keep_float_outputs=["no_such_tensor"])
    assert not (tmp_path / "bad_prep.onnx").exists()
