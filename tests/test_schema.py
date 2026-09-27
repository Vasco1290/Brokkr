"""Checks for brokkr.schema: the unified result format, its checker, and the schema-1 converter.

The numbers in these records are made up for the tests; they are never results.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from brokkr.results import sha256_of
from brokkr.schema import (
    check_build_record,
    check_record,
    condition,
    condition_label,
    from_v1,
    load_measurement,
    load_measurements,
    make_measurement,
    metric,
    non_default_settings,
    save_measurement,
)

COMMIT = "a" * 40
FINGERPRINT = {"cpu_model": "test CPU", "board_model": None, "architecture": "AMD64",
               "os": "Windows", "os_release": "11",
               "packages": {"onnxruntime": "1.23.2"}, "git": {"commit": COMMIT, "dirty": False}}
MODEL = {"name": "tiny", "weights": "TINY_V1", "licence": "Apache-2.0 (test)"}
RUNTIME = {"name": "onnxruntime", "version": "1.23.2", "execution_provider": "CPUExecutionProvider",
           "threads": 4}
DATA = {"dataset": "fake", "split": "test", "n_images": 100, "licence": "test licence"}


def accuracy_record(**changes) -> dict:
    record = make_measurement("accuracy", MODEL, "fp32", RUNTIME, "laptop", FINGERPRINT, DATA,
                              condition("fog", "imagenet-c", 3), {"top1": metric(0.5, [0.4, 0.6])},
                              {"batch_size": 32})
    record.update(changes)
    return record


def test_a_complete_record_passes():
    assert check_record(accuracy_record()) == []


def test_condition_labels_always_name_the_suite():
    assert condition_label(condition()) == "clean"
    assert condition_label(condition("fog", "brokkr", 3)) == "fog (Brokkr) s3"
    assert condition_label(condition("defocus_blur", "imagenet-c", 5)) == "defocus blur (ImageNet-C) s5"


@pytest.mark.parametrize("change, expected", [
    ({"model": {"name": "tiny", "weights": "TINY_V1", "licence": ""}}, "no licence"),
    ({"data": {**DATA, "licence": None}}, "no licence"),
    ({"source": "someone"}, "hard rule 7"),
    ({"condition": condition("fog", None, 3)}, "must name its suite"),
    ({"condition": condition("clean", "brokkr", 0)}, "clean condition"),
    ({"condition": condition("fog", "brokkr", 6)}, "severity 1-5"),
    ({"metrics": {"top1": metric(0.5)}}, "needs its 95% interval"),
    ({"metrics": {"top1": metric(0.5, [0.6, 0.4])}}, "malformed 95% interval"),
    ({"metrics": {"top1": metric(float("nan"), [0.4, 0.6])}}, "finite number"),
    ({"runtime": {**RUNTIME, "threads": 0}}, "threads"),
    ({"code": {"commit": "abc", "dirty": False}}, "git commit"),
    ({"arrays": {"file": "x.npz", "sha256": "not-a-checksum"}}, "SHA-256"),
])
def test_each_rule_is_enforced(change, expected):
    problems = check_record(accuracy_record(**change))
    assert any(expected in p for p in problems), problems


def test_coverage_is_never_accepted_without_set_size():
    record = accuracy_record(kind="conformal", metrics={"coverage": metric(0.9, [0.89, 0.91])})
    assert any("mean_set_size" in p for p in check_record(record))


def test_speed_needs_enough_runs_and_no_images():
    speed = make_measurement("speed", MODEL, "fp32", RUNTIME, "laptop", FINGERPRINT, None, None,
                             {k: metric(1.0) for k in ("p50_ms", "p95_ms", "p99_ms")},
                             {"warmup_runs": 20, "timed_runs": 100})
    assert check_record(speed) == []
    speed["settings"]["warmup_runs"] = 10
    assert any("warm-up" in p for p in check_record(speed))
    speed["settings"]["warmup_runs"] = 20
    speed["data"] = DATA
    assert any("random input" in p for p in check_record(speed))


def test_device_label_must_match_the_fingerprint():
    record = accuracy_record()
    record["device"]["label"] = "raspberry-pi-5"
    assert any("not a Raspberry Pi 5" in p for p in check_record(record))
    record["device"]["fingerprint"] = {**FINGERPRINT, "board_model": "Raspberry Pi 5 Model B Rev 1.0"}
    assert check_record(record) == []
    record["device"]["label"] = "laptop"
    assert any("names a board" in p for p in check_record(record))
    record["device"] = {"label": "cloud-arm", "fingerprint": FINGERPRINT}   # x86, so not cloud ARM
    assert any("not an ARM machine" in p for p in check_record(record))


def test_the_os_must_be_recorded():
    record = accuracy_record()
    record["device"]["fingerprint"] = {k: v for k, v in FINGERPRINT.items() if k != "os"}
    assert any("must record the OS" in p for p in check_record(record))


def test_a_diagnostic_needs_its_script_split_and_count():
    record = make_measurement("diagnostic", MODEL, "int8", RUNTIME, "laptop", FINGERPRINT,
                              {**DATA, "split": "tuning"}, condition(),
                              {"median_sqnr_db": metric(3.0)}, {"script": "scripts/26_x.py"})
    assert check_record(record) == []
    record["settings"] = {}
    assert any("settings.script" in p for p in check_record(record))
    record["settings"] = {"script": "scripts/26_x.py"}
    record["data"] = {**DATA, "n_images": 0}
    assert any("number of images" in p for p in check_record(record))


def test_save_refuses_a_failing_record(tmp_path):
    with pytest.raises(ValueError, match="schema check"):
        save_measurement(accuracy_record(source="someone"), tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
    assert save_measurement(accuracy_record(), tmp_path / "good.json").exists()


def test_load_measurement_checks_the_arrays(tmp_path):
    np.savez_compressed(tmp_path / "scores.npz", logits=np.zeros((2, 3), np.float32))
    record = accuracy_record(arrays={"file": "scores.npz", "sha256": sha256_of(tmp_path / "scores.npz")})
    save_measurement(record, tmp_path / "r.json")
    loaded, arrays = load_measurement(tmp_path / "r.json")
    assert loaded == record and arrays["logits"].shape == (2, 3)
    np.savez_compressed(tmp_path / "scores.npz", logits=np.ones((2, 3), np.float32))  # changed afterwards
    with pytest.raises(ValueError, match="checksum"):
        load_measurement(tmp_path / "r.json")


# ---- schema 1 -> schema 2 ----

V1_MACHINE = {**FINGERPRINT}
V1_ACCURACY = {
    "schema_version": 1, "kind": "accuracy", "source": "brokkr", "model": "tiny", "precision": "int8",
    "settings": {"n_images": 100, "num_threads": 4, "dataset": "fake", "split": "test",
                 "corruption": "darkness", "severity": 5},
    "metrics": {"top1": 0.5, "top1_ci95": [0.4, 0.6], "top1_range_over_tie_breaks": [0.5, 0.51]},
    "raw_arrays": {"file": "run.npz", "sha256": "b" * 64},
    "machine": V1_MACHINE,
}
V1_CONFORMAL = {
    "schema_version": 1, "kind": "conformal", "source": "brokkr", "model": "tiny", "precision": "int8",
    "settings": {"dataset": "fake", "split": "test", "n_images": 100, "source_result": "run.json",
                 "source_arrays_sha256": "b" * 64},
    "metrics": {"coverage": 0.9, "coverage_ci95": [0.85, 0.95], "mean_set_size": 2.0,
                "mean_set_size_ci95": [1.9, 2.1]},
    "machine": V1_MACHINE,
}
MODELS = {"tiny": {"weights": "TINY_V1", "licence": "Apache-2.0 (test)"}}
DATASETS = {"fake": {"licence": "test licence"}}


def test_v1_records_convert_and_pass():
    by_file = {"run.json": V1_ACCURACY}
    acc = from_v1(V1_ACCURACY, "run.json", MODELS, DATASETS, by_file)
    assert check_record(acc) == []
    assert acc["condition"] == {"corruption": "darkness", "suite": "brokkr", "severity": 5}
    assert acc["metrics"]["top1"] == {"value": 0.5, "ci95": [0.4, 0.6]}
    assert acc["settings"]["details"]["top1_range_over_tie_breaks"] == [0.5, 0.51]
    assert acc["data"]["licence"] == "test licence"

    conf = from_v1(V1_CONFORMAL, "run_conformal.json", MODELS, DATASETS, by_file)
    assert check_record(conf) == []
    assert conf["runtime"]["threads"] == 4                   # taken from the run it was computed from
    assert conf["condition"] == condition()                  # no corruption recorded = clean
    assert conf["derived_from"][1] == {"file": "run.json", "arrays_sha256": "b" * 64}


def test_v1_conversion_does_not_invent_a_licence():
    acc = from_v1(V1_ACCURACY, "run.json", {}, DATASETS, {})
    assert any("no licence" in p for p in check_record(acc))


def test_decision_records_are_skipped_not_converted():
    choice = {"schema_version": 1, "kind": "choice", "model": "tiny"}
    assert from_v1(choice, "choice.json", MODELS, DATASETS, {}) is None


def test_missing_source_result_is_an_error():
    with pytest.raises(ValueError, match="source result"):
        from_v1(V1_CONFORMAL, "run_conformal.json", MODELS, DATASETS, {})


def test_load_measurements_reads_both_versions(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps(V1_ACCURACY))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "run_conformal.json").write_text(json.dumps(V1_CONFORMAL))
    (tmp_path / "new.json").write_text(json.dumps(accuracy_record()))
    (tmp_path / "choice.json").write_text(json.dumps({"schema_version": 1, "kind": "choice"}))
    measurements, skipped = load_measurements(tmp_path, MODELS, DATASETS)
    assert len(measurements) == 3
    assert [kind for _, kind in skipped] == ["choice"]
    assert all(check_record(r) == [] for _, r in measurements)


def test_converting_does_not_change_the_input():
    before = copy.deepcopy(V1_ACCURACY)
    from_v1(V1_ACCURACY, "run.json", MODELS, DATASETS, {})
    assert V1_ACCURACY == before


@pytest.mark.skipif(not Path("results/accuracy").exists(), reason="Stage 1-3 results not on this machine")
def test_every_real_stage1_to_3_measurement_passes():
    from brokkr.datasets import DATASETS as REAL_DATASETS
    from brokkr.export import MODELS as REAL_MODELS
    models = {n: {"weights": str(s["weights"]), "licence": s["licence"]} for n, s in REAL_MODELS.items()}
    measurements, _ = load_measurements("results", models, REAL_DATASETS)
    assert measurements
    assert [p for p, r in measurements if check_record(r)] == []


# ---- model build records ----

def build_record(**changes):
    record = {"model": "tiny", "precision": "int8_percentile99.99", "licence": "Apache-2.0 (test)",
              "settings": {"calibration_method": "percentile99.99", "skip_symbolic_shape": False,
                           "calibration": {"split": "int8_calibration", "group_images": 128}},
              "build": {"checks": {"agreement >= 20%": True}},
              "machine": {"git": {"commit": COMMIT, "dirty": False}}}
    record.update(changes)
    return record


def test_a_clean_passing_build_is_usable():
    assert check_build_record(build_record(), {"skip_symbolic_shape"}) == ("usable", [])


def test_a_failed_sanity_check_is_reported_not_hidden():
    status, problems = check_build_record(build_record(build={"checks": {"agreement >= 20%": False}}), set())
    assert status == "failed" and problems == []


def test_older_records_are_judged_by_their_own_check():
    export = {**build_record(), "pytorch_vs_onnx": {"max_abs_diff": 1e-6, "top1_agreement": 1.0}}
    del export["build"]
    assert check_build_record(export, set())[0] == "usable"
    stage3 = {**build_record(), "sanity_check": {"top1_agreement_with_fp32": 0.1}}
    del stage3["build"]
    assert check_build_record(stage3, set())[0] == "failed"
    unchecked = build_record()
    del unchecked["build"]
    assert check_build_record(unchecked, set())[0] == "no sanity check"


def test_a_dirty_build_or_missing_licence_is_a_problem():
    dirty = build_record(machine={"git": {"commit": COMMIT, "dirty": True}})
    assert any("clean commit" in p for p in check_build_record(dirty, set())[1])
    assert any("licence" in p for p in check_build_record(build_record(licence=""), set())[1])


def test_an_option_the_code_had_must_be_stated():
    record = build_record()
    del record["settings"]["skip_symbolic_shape"]
    assert check_build_record(record, set())[1] == []                       # built before the option
    assert any("does not state it" in p for p in check_build_record(record, {"skip_symbolic_shape"})[1])


def test_non_default_settings_are_shown():
    assert non_default_settings(build_record()) == {}
    record = build_record()
    record["settings"]["skip_symbolic_shape"] = True
    record["settings"]["calibration"]["group_images"] = 64
    assert non_default_settings(record) == {"skip_symbolic_shape": True, "calibration group_images": 64}
