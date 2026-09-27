"""M2's measures and verdict rules (brokkr.m2) on made-up inputs whose right answer is known.

The numbers here are invented for the tests; they are never results.
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.qdq_loss_debug import compute_signal_to_quantization_noice_ratio

from brokkr import m2
from brokkr.levels import sqnr_db_per_image


def tiny_qdq_model() -> onnx.ModelProto:
    """input -> Q -> DQ -> Relu -> Q -> DQ -> output, plus a quantized weight (must be ignored)."""
    init = [numpy_helper.from_array(np.array(0.1, np.float32), "s1"),
            numpy_helper.from_array(np.array(128, np.uint8), "z1"),
            numpy_helper.from_array(np.array(0.05, np.float32), "s2"),
            numpy_helper.from_array(np.array(0, np.uint8), "z2"),
            numpy_helper.from_array(np.ones(3, np.float32), "w")]
    nodes = [helper.make_node("QuantizeLinear", ["images", "s1", "z1"], ["images_q"]),
             helper.make_node("DequantizeLinear", ["images_q", "s1", "z1"], ["images_dq"]),
             helper.make_node("QuantizeLinear", ["w", "s1", "z1"], ["w_q"]),  # a weight, not an activation
             helper.make_node("Relu", ["images_dq"], ["relu"]),
             helper.make_node("QuantizeLinear", ["relu", "s2", "z2"], ["relu_q"]),
             helper.make_node("DequantizeLinear", ["relu_q", "s2", "z2"], ["out"])]
    graph = helper.make_graph(nodes, "tiny",
                              [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3])],
                              [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3])], init)
    return helper.make_model(graph)


def test_activation_quantizers_in_graph_order_without_weights():
    found = m2.activation_quantizers(tiny_qdq_model())
    assert [f["tensor"] for f in found] == ["images", "relu"]
    assert [f["dequantized"] for f in found] == ["images_dq", "out"]
    assert float(found[1]["scale"]) == pytest.approx(0.05) and found[1]["zero_point"].dtype == np.uint8


def test_sqnr_matches_onnxruntime_and_the_per_image_version():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3, 5, 5)).astype(np.float32)
    y = x + rng.normal(scale=0.1, size=x.shape).astype(np.float32)
    s2, n2 = m2.squared_norms(x, y)
    assert m2.sqnr_db(s2, n2) == pytest.approx(sqnr_db_per_image(x, y))
    assert m2.pooled_sqnr_db(s2, n2) == pytest.approx(compute_signal_to_quantization_noice_ratio([x], [y]))
    # ONNX Runtime's guard: identical values give a finite (very large) SQNR, zeros give 0 dB.
    assert np.isfinite(m2.sqnr_db(*m2.squared_norms(x, x))).all()
    assert m2.sqnr_db(0.0, 0.0) == 0.0


def test_local_rounding_uses_fake_quantize_with_the_clip():
    x = np.array([[0.0, 0.1, 0.2, 100.0]], np.float32)  # 100 is far beyond the 8-bit range at scale 0.1
    s2, n2 = m2.local_rounding(x, np.float32(0.1), np.uint8(0))
    assert n2[0] == pytest.approx((100.0 - 25.5) ** 2, rel=1e-5)  # clipped at level 255 = 25.5


def test_input_level_ratio():
    fine = np.tile(np.arange(10, dtype=np.float32) / 10, (1, 3, 1)).reshape(1, 3, 10)  # 10 values each
    assert m2.input_level_ratio(fine, np.float32(0.01), np.uint8(0)).tolist() == [1.0]    # all stay distinct
    # 0.0..0.9 / 0.5 rounds to levels 0, 1 and 2: 3 levels for 10 values.
    assert m2.input_level_ratio(fine, np.float32(0.5), np.uint8(0)).tolist() == [pytest.approx(0.3)]


def test_early_block_and_extra_error():
    assert m2.early_count(125) == 13 and m2.early_count(10) == 1 and m2.early_count(125, 0.05) == 7
    rng = np.random.default_rng(1)
    clean = 30 + rng.normal(0, 0.5, (128, 21))
    early_worse = clean.copy()
    early_worse[:, 1:3] -= 5  # 20 tensors after the input: early block = 2 tensors, 5 dB worse
    s = m2.extra_error(clean, early_worse)
    assert s["early_tensors"] == 2 and s["rest_tensors"] == 18
    assert s["E_early"] == pytest.approx(5) and s["E_rest"] == pytest.approx(0, abs=1e-9)
    assert s["E_early_ci95"][0] > 0 and s["early_minus_rest_ci95"][0] > 0
    assert m2.m2a_model(s) == {"supports": True, "rejects": False}

    everywhere = clean - 5  # worse everywhere: not concentrated early, E_early <= E_rest
    assert m2.m2a_model(m2.extra_error(clean, everywhere)) == {"supports": False, "rejects": True}
    better_early = clean.copy()
    better_early[:, 1:3] += 5  # interval entirely below zero counts as "not above zero" -> rejects
    assert m2.m2a_model(m2.extra_error(clean, better_early))["rejects"]
    tiny = clean - 0.5  # never reaches 1.0 dB: "no extra error", cannot support
    t = m2.extra_error(clean, tiny)
    assert t["no_extra_error"] and not m2.m2a_model(t)["supports"]


def test_verdict_counts():
    assert m2.rejects_needed(8) == 5 and m2.rejects_needed(7) == 4
    six = {f"m{i}": {"supports": i < 6, "rejects": False} for i in range(8)}
    assert m2.verdict(six, 5)["verdict"] == "SUPPORTS"
    five = {f"m{i}": {"supports": i < 5, "rejects": False} for i in range(8)}
    assert m2.verdict(five, 5)["verdict"] == "INCONCLUSIVE"
    rej = {f"m{i}": {"supports": False, "rejects": i < 4} for i in range(7)}
    assert m2.verdict(rej, 4)["verdict"] == "REJECTS" and m2.verdict(rej, 5)["verdict"] == "INCONCLUSIVE"


def test_ratio_change_band_is_inclusive_and_below_is_strict():
    clean = np.zeros(1)  # one image, so the mean is exactly -0.05 (no rounding in a sum)
    at_edge = m2.ratio_change(clean, clean - 0.05)
    assert at_edge["ci95"] == [-0.05, -0.05] and at_edge["supports"] and not at_edge["rejects"]
    below = m2.ratio_change(clean, clean - 0.2)
    assert below["rejects"] and not below["supports"]
