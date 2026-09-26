"""Checks for brokkr.levels (the counts are on small made-up arrays, not results)."""

import numpy as np
import onnx
from onnx import TensorProto, helper

from brokkr.levels import activation_quantize_outputs, distinct_levels, expose


def test_distinct_levels_counts_each_image_separately():
    values = np.array([[[0, 0, 5], [5, 5, 5]],        # image 0 uses levels 0 and 5
                       [[1, 2, 3], [4, 255, 255]]],    # image 1 uses 1, 2, 3, 4, 255
                      dtype=np.uint8)
    assert distinct_levels(values).tolist() == [2, 5]


def test_int8_values_are_counted_like_uint8():
    values = np.array([[-128, 127, 127, 0]], dtype=np.int8)
    assert distinct_levels(values).tolist() == [3]


def test_all_256_levels_can_be_counted():
    assert distinct_levels(np.arange(256, dtype=np.uint8)[None, :]).tolist() == [256]


def tiny_qdq_model():
    """input -> QuantizeLinear (activation) -> DequantizeLinear -> output; a weight also quantized."""
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1])
    zero = helper.make_tensor("zero", TensorProto.UINT8, [], [0])
    weight = helper.make_tensor("w", TensorProto.FLOAT, [2], [1.0, 2.0])
    nodes = [helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["x_q"]),
             helper.make_node("DequantizeLinear", ["x_q", "scale", "zero"], ["y"]),
             helper.make_node("QuantizeLinear", ["w", "scale", "zero"], ["w_q"])]
    graph = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
                              [scale, zero, weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_only_activation_quantizers_are_found_and_exposed(tmp_path):
    model = tiny_qdq_model()
    assert activation_quantize_outputs(model) == ["x_q"]  # the weight's quantizer is excluded
    onnx.save(model, tmp_path / "m.onnx")
    exposed = onnx.load(expose(tmp_path / "m.onnx", tmp_path / "e.onnx", ["x_q"]))
    assert [o.name for o in exposed.graph.output] == ["y", "x_q"]  # normal output stays first
