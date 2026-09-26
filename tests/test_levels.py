"""Checks for brokkr.levels (the counts are on small made-up arrays, not results)."""

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from brokkr.levels import (
    SATURATION,
    activation_quantize_outputs,
    distinct_levels,
    expose,
    fake_quantize,
    sqnr_db_per_image,
)


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


def qdq_session(scale, zero_point, dtype):
    """ONNX Runtime running QuantizeLinear then DequantizeLinear, to compare fake_quantize with."""
    import onnxruntime as ort
    elem = TensorProto.UINT8 if dtype == np.uint8 else TensorProto.INT8
    s = helper.make_tensor("s", TensorProto.FLOAT, [], [scale])
    z = helper.make_tensor("z", elem, [], [zero_point])
    nodes = [helper.make_node("QuantizeLinear", ["x", "s", "z"], ["q"]),
             helper.make_node("DequantizeLinear", ["q", "s", "z"], ["y"])]
    graph = helper.make_graph(nodes, "qdq", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])], [s, z])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # the onnx package writes a newer file version than this ONNX Runtime reads
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("zero_point, dtype", [(128, np.uint8), (3, np.uint8), (0, np.int8), (-20, np.int8)])
def test_fake_quantize_equals_onnx_runtime_including_clipping_and_ties(zero_point, dtype):
    scale = 0.05
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 5, 10_000),                 # many values beyond the 8-bit range
                        np.arange(-300, 300) * scale / 2,          # exact half-way ties
                        [1e6, -1e6, 0.0]]).astype(np.float32)
    expected = qdq_session(scale, zero_point, dtype).run(None, {"x": x})[0]
    ours = fake_quantize(x, scale, np.array(zero_point, dtype=dtype))
    assert np.array_equal(ours, expected)
    top = (SATURATION[np.dtype(dtype)][1] - zero_point) * scale
    assert (x > top).any() and np.array_equal(ours[x > top], np.full((x > top).sum(), top, np.float32))


def test_sqnr_per_image_matches_the_formula():
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    y = np.array([[3.0, 3.0], [1.0, 0.0]])        # error norm 1 vs signal 5; second image exact
    out = sqnr_db_per_image(x, y)
    assert np.isclose(out[0], 20 * np.log10(5.0)) and out[1] > 300


@pytest.mark.skipif(not Path("models/mobilenet_v3_large_int8_percentile99.99.onnx").exists(),
                    reason="models not built on this machine")
def test_fake_quantize_equals_every_real_qdq_pair_of_a_percentile_model(tmp_path):
    import onnxruntime as ort
    from onnx import numpy_helper
    model = onnx.load("models/mobilenet_v3_large_int8_percentile99.99.onnx")
    stored = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    quant = [n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.input[0] not in stored]
    dequant = {n.input[0]: n.output[0] for n in model.graph.node if n.op_type == "DequantizeLinear"}
    names = []
    for n in quant:
        names += [n.input[0], dequant[n.output[0]]]
    existing = {o.name for o in model.graph.output}
    for name in names:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
            existing.add(name)
    onnx.save(model, tmp_path / "m.onnx")
    session = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"])
    wanted = [o.name for o in session.get_outputs()]
    images = np.random.default_rng(0).normal(0, 1.5, (2, 3, 224, 224)).astype(np.float32)
    values = dict(zip(wanted, session.run(None, {"images": images}), strict=True))
    clipped = 0
    for n in quant:
        scale, zp = stored[n.input[1]], stored[n.input[2]]
        before, after = values[n.input[0]], values[dequant[n.output[0]]]
        assert np.array_equal(fake_quantize(before, scale, zp), after), n.input[0]
        clipped += int((np.abs(before - after) > scale).any())   # an error bigger than one step = a clip
    assert clipped > 0   # the test really covers Percentile's clipping
