"""brokkr_edge.label_build: the label pieces the study and the user path share (made-up numbers, tests
only).

That the study's labels come out unchanged is shown by remaking them and comparing with the released ones
(scripts/48_compare_labels.py); these tests pin the rules of each piece.
"""

import numpy as np

from brokkr_edge.label_build import (
    assemble,
    build_entry,
    derived_measurements,
    envelope_rows,
    failure_block,
    general_limits,
    measurement,
)
from brokkr_edge.label_schema import REQUIRED_KEYS

REF_ID, LAB_ID = "m#fp32:aaaaaaaaaaaa", "m#int8:bbbbbbbbbbbb"
BUILDS = {
    "reference": build_entry(REF_ID, "reference", "FP32", "fp32", {}, "a" * 64, 10, "usable", None),
    "labelled": build_entry(LAB_ID, "labelled", "INT8", "int8", {}, "b" * 64, 5, "usable", None),
}
CONDS = ["clean", "brokkr/fog/3"]


def made_up_correct(seed):
    return np.random.default_rng(seed).integers(0, 2, 200).astype(np.float64)


def test_measurement_names_the_second_runtime_only_when_it_differs():
    args = ("damage_drop", "b", "d", "c", "h", "rt1", 0.1, [0.0, 0.2], 10, {}, [])
    same, other = measurement(*args, paired_runtime="rt1"), measurement(*args, paired_runtime="rt2")
    assert "paired_runtime_id" not in same["settings"]
    assert other["settings"]["paired_runtime_id"] == "rt2"
    assert measurement("mean_set_size", "b", "d", "c", "h", "r", 2.0, None, 10, {}, [])["unit"] == "classes"


def test_derived_measurements_order_and_pairing():
    correct = {(r, c): made_up_correct(i) for i, (r, c) in enumerate([(r, c) for r in BUILDS for c in CONDS])}
    sources = {key: {"file": f"{key}", "sha256": "c" * 64, "field": "logits, labels"} for key in correct}
    runtimes = {key: "rt" for key in correct}
    out = derived_measurements(BUILDS, {"reference": "usable", "labelled": "usable"}, CONDS, correct, sources,
                               runtimes, "d", "h")
    assert [(m["metric"], m["build_id"], m["condition_id"]) for m in out] == [
        ("damage_drop", BUILDS["reference"]["build_id"], "brokkr/fog/3"),
        ("damage_drop", BUILDS["labelled"]["build_id"], "brokkr/fog/3"),
        ("shrinking_cost", BUILDS["labelled"]["build_id"], "clean"),
        ("shrinking_cost", BUILDS["labelled"]["build_id"], "brokkr/fog/3"),
    ]
    drop = out[0]
    expected = correct[("reference", "brokkr/fog/3")].mean() - correct[("reference", "clean")].mean()
    assert abs(drop["value"] - expected) < 1e-12 and drop["n_items"] == 200


def test_a_failed_build_has_no_derived_measurements_and_failed_rows():
    failure = failure_block("agreement", 0.03, 0.2, 256, "x", [])
    failed = dict(BUILDS, labelled=build_entry(LAB_ID, "labelled", "INT8", "int8", {}, "b" * 64, 5, "failed",
                                               failure))
    status = {"reference": "usable", "labelled": "failed"}
    correct = {("reference", c): made_up_correct(i) for i, c in enumerate(CONDS)}
    sources = {key: {"file": "f", "sha256": "c" * 64, "field": "x"} for key in correct}
    out = derived_measurements(failed, status, CONDS, correct, sources, {k: "rt" for k in correct}, "d", "h")
    assert {m["build_id"] for m in out} == {BUILDS["reference"]["build_id"]}
    copied = [measurement(m, BUILDS["reference"]["build_id"], "d", c, "h", "rt", v, ci, 200, {}, [])
              for c in CONDS for m, v, ci in (("top1", 0.5, [0.4, 0.6]), ("coverage", 0.9, [0.85, 0.95]))]
    rows = envelope_rows(failed, status, CONDS[1:], copied + out, "h")
    labelled = [r for r in rows if r["build_id"] == failed["labelled"]["build_id"]]
    assert [r["state"] for r in labelled] == ["INT8 build failed"]
    assert labelled[0]["why"] == ["build check failed: agreement"] and labelled[0]["line_states"] is None


def test_general_limits_takes_the_machine_sentence_as_given():
    lines = general_limits("cpu", "os 1", 12, "where it was measured")
    assert lines[2] == "Measured on one machine: cpu, os 1." and "With 12 conditions" in lines[3]
    assert lines[5] == "where it was measured" and len(lines) == 7


def test_assemble_keeps_schema_order():
    label = assemble(source={}, generated={}, model={}, builds=BUILDS, hardware=[], runtimes=[], datasets=[],
                     conditions=[], checks={}, measurements=[], rows=[], speed=[], details={}, limits=[],
                     licences={}, sources=[])
    assert [k for k in label if k in REQUIRED_KEYS] == list(REQUIRED_KEYS)
    assert label["label_id"] == BUILDS["labelled"]["build_id"]
