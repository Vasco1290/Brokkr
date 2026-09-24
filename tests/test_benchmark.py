"""Checks for brokkr.benchmark and brokkr.results.

Uses a tiny one-operation ONNX model built by hand, so the tests run in about a second.
"""

import onnx
import pytest
from onnx import TensorProto, helper

from brokkr.benchmark import benchmark_latency, combine_sessions, summarise
from brokkr.fingerprint import machine_fingerprint
from brokkr.results import load_records, make_record, save_record


@pytest.fixture
def tiny_model(tmp_path):
    """An ONNX model that just applies ReLU to a (1, 3, 8, 8) input."""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "tiny", [x], [y])
    # ir_version 10: the newest onnx library writes a file format ONNX Runtime can't read yet.
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = tmp_path / "tiny.onnx"
    onnx.save(model, str(path))
    return path


def test_benchmark_records_settings_and_all_runs(tiny_model):
    result = benchmark_latency(tiny_model, input_shape=(1, 3, 8, 8), num_threads=2)
    assert result["settings"]["num_threads"] == 2
    assert result["settings"]["warmup_runs"] >= 20
    assert len(result["raw"]["latencies_ms"]) == result["settings"]["timed_runs"] >= 100


def test_percentiles_are_ordered(tiny_model):
    m = benchmark_latency(tiny_model, input_shape=(1, 3, 8, 8), num_threads=1)["metrics"]
    assert 0 < m["min_ms"] <= m["p50_ms"] <= m["p95_ms"] <= m["p99_ms"] <= m["max_ms"]


def test_too_few_runs_is_refused(tiny_model):
    with pytest.raises(ValueError):
        benchmark_latency(tiny_model, input_shape=(1, 3, 8, 8), runs=10)
    with pytest.raises(ValueError):
        benchmark_latency(tiny_model, input_shape=(1, 3, 8, 8), warmup=5)


def test_summarise_on_known_numbers():
    m = summarise(list(range(1, 101)))  # 1, 2, ..., 100
    assert m["min_ms"] == 1 and m["max_ms"] == 100
    assert m["p50_ms"] == pytest.approx(50.5)
    assert m["mean_ms"] == pytest.approx(50.5)


def test_combine_sessions_uses_median_and_reports_spread():
    # Three fake sessions whose p50s are 10, 12, and 20 ms.
    sessions = [
        {"settings": {"num_threads": 1, "timed_runs": 100},
         "metrics": {"mean_ms": p, "p50_ms": p, "p95_ms": p + 1, "p99_ms": p + 2},
         "raw": {"latencies_ms": [p] * 100}}
        for p in (10.0, 12.0, 20.0)
    ]
    result = combine_sessions(sessions)
    m = result["metrics"]
    assert m["p50_ms"] == 12.0  # median, so the slow 20 ms session doesn't drag it up
    assert (m["p50_ms_min"], m["p50_ms_max"]) == (10.0, 20.0)
    assert m["p50_spread_pct"] == pytest.approx((20 - 10) / 12 * 100)
    # IQR of [10, 12, 20]: 25th percentile 11, 75th percentile 16
    assert m["p50_iqr_pct"] == pytest.approx((16 - 11) / 12 * 100)
    assert result["settings"]["sessions"] == 3
    assert len(result["raw"]["sessions"]) == 3


def test_iqr_ignores_one_unlucky_session():
    p50s = [5.0, 5.1, 5.0, 5.2, 5.1, 5.0, 5.1, 5.2, 5.0, 11.0]  # one hot session
    sessions = [{"settings": {}, "metrics": {"mean_ms": p, "p50_ms": p, "p95_ms": p, "p99_ms": p},
                 "raw": {"latencies_ms": [p]}} for p in p50s]
    m = combine_sessions(sessions)["metrics"]
    assert m["p50_spread_pct"] > 100  # full spread is dominated by the one bad session
    assert m["p50_iqr_pct"] < 5  # the typical sessions agreed closely


def test_combine_needs_at_least_two_sessions(tiny_model):
    one = benchmark_latency(tiny_model, input_shape=(1, 3, 8, 8), num_threads=1)
    with pytest.raises(ValueError):
        combine_sessions([one])


def test_record_round_trip(tiny_model, tmp_path):
    result = benchmark_latency(tiny_model, input_shape=(1, 3, 8, 8), num_threads=1)
    record = make_record("speed", "tiny", "fp32", result, machine_fingerprint())
    save_record(record, tmp_path / "out" / "tiny.json")

    [loaded] = load_records(tmp_path / "out")
    assert loaded["source"] == "brokkr"
    assert loaded["metrics"] == record["metrics"]


def test_incomplete_record_is_refused(tmp_path):
    with pytest.raises(ValueError):
        save_record({"kind": "speed"}, tmp_path / "bad.json")


def test_pinning_to_one_cpu_then_back(tiny_model):
    import os

    from brokkr.benchmark import pin_to_cpus
    try:
        pin_to_cpus([0])
        result = benchmark_latency(tiny_model, input_shape=(1, 3, 8, 8), num_threads=1)
        assert result["metrics"]["p50_ms"] > 0
    finally:
        pin_to_cpus(list(range(os.cpu_count())))  # don't leave the test process pinned


def test_arrays_saved_with_checksum_and_reproduce_metrics(tmp_path):
    import numpy as np

    from brokkr.accuracy import accuracy_from_logits
    from brokkr.results import load_arrays, save_arrays

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((50, 1000)).astype(np.float32)
    labels = rng.integers(0, 1000, 50)
    result = accuracy_from_logits(logits, labels)
    record = make_record("accuracy", "m", "fp32", {"settings": {}, **result}, machine_fingerprint())
    json_path = tmp_path / "r.json"
    save_arrays(record, json_path, logits=logits, labels=labels)
    save_record(record, json_path)

    arrays = load_arrays(json_path)
    assert np.array_equal(arrays["logits"], logits)  # exact, not approximately equal
    again = accuracy_from_logits(arrays["logits"], arrays["labels"])
    assert again["metrics"] == result["metrics"]
    assert record["raw_arrays"]["arrays"]["logits"]["shape"] == [50, 1000]


def test_edited_array_file_is_detected(tmp_path):
    import numpy as np

    from brokkr.results import load_arrays, save_arrays

    record = make_record("accuracy", "m", "fp32", {"settings": {}, "metrics": {}},
                         machine_fingerprint())
    json_path = tmp_path / "r.json"
    save_arrays(record, json_path, logits=np.zeros((2, 3), dtype=np.float32))
    save_record(record, json_path)
    np.savez_compressed(tmp_path / "r.npz", logits=np.ones((2, 3), dtype=np.float32))  # tamper
    with pytest.raises(ValueError):
        load_arrays(json_path)
